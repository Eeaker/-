const api = require('../../utils/api.js');
const app = getApp();

Page({
  data: {
    baseUrl: '',
    loading: true,
    refreshing: false,
    list: [],
    page: 1,
    pageSize: 10,
    totalPages: 1,
    selectedTaskId: null,
    detailLoading: false,
    detail: null,
    detailTracking: {},
    detailMetrics: {},
    detailShotPrediction: null,
    detailShotView: {
      label: '未知',
      detail: '未提供篮筐标定参数或轨迹不足',
    },
    videoUrl: '',
    pointSamples: [],
    initialTaskId: null,
    texts: {
      title: '\u8f68\u8ff9\u5386\u53f2',
      subtitle: '\u652f\u6301\u56de\u653e\u8f68\u8ff9\u89c6\u9891\u5e76\u70b9\u51fb\u8f68\u8ff9\u70b9\u8df3\u8f6c',
      loading: '\u52a0\u8f7d\u4e2d...',
      loadingDetail: '\u52a0\u8f7d\u4efb\u52a1\u8be6\u60c5...',
      selectHint: '\u8bf7\u9009\u62e9\u4efb\u52a1\u67e5\u770b\u8be6\u60c5',
      taskPrefix: '\u4efb\u52a1 #',
      detected: '\u68c0\u6d4b\u70b9',
      coverage: '\u8986\u76d6\u7387',
      loadMore: '\u52a0\u8f7d\u66f4\u591a',
      detectorMode: '\u68c0\u6d4b\u6a21\u5f0f',
      detectedPoints: '\u68c0\u6d4b\u70b9',
      predictedPoints: '\u9884\u6d4b\u70b9',
      continuity: '\u8fde\u7eed\u6027',
      shotPrediction: '\u6295\u7bee\u5224\u5b9a',
      videoPending: '\u8f68\u8ff9\u89c6\u9891\u6682\u672a\u751f\u6210',
      pointDetected: '\u68c0\u6d4b',
      pointPredicted: '\u9884\u6d4b',
      speedLabel: '\u901f\u5ea6',
      frameLabel: '\u5e27',
      refreshFailed: '\u8f68\u8ff9\u5386\u53f2\u52a0\u8f7d\u5931\u8d25',
      detailFailed: '\u8f68\u8ff9\u8be6\u60c5\u52a0\u8f7d\u5931\u8d25'
    },
  },

  pollTimer: null,

  onLoad(options) {
    this.setData({ baseUrl: app.globalData.baseUrl });
    const initialTaskId = Number(options?.taskId || 0);
    if (initialTaskId > 0) {
      this.setData({ initialTaskId });
    }
  },

  onShow() {
    this.refreshAll();
    this.startProcessingPoll();
  },

  onHide() {
    this.stopProcessingPoll();
  },

  onUnload() {
    this.stopProcessingPoll();
  },

  onPullDownRefresh() {
    this.refreshAll().finally(() => wx.stopPullDownRefresh());
  },

  normalizeListResponse(payload) {
    if (Array.isArray(payload)) {
      return { items: payload, page: 1, totalPages: 1 };
    }
    return {
      items: Array.isArray(payload.items) ? payload.items : [],
      page: Number(payload?.pagination?.page || 1),
      totalPages: Number(payload?.pagination?.total_pages || 1),
    };
  },

  statusLabel(status) {
    if (status === 'completed') return '\u5df2\u5b8c\u6210';
    if (status === 'processing') return '\u5904\u7406\u4e2d';
    if (status === 'failed') return '\u5931\u8d25';
    if (status === 'cancelled') return '\u5df2\u53d6\u6d88';
    return status || '\u7b49\u5f85\u4e2d';
  },

  async loadPage(targetPage = 1, append = false) {
    const resp = await api.get('/api/trajectory/history', {
      page: targetPage,
      page_size: this.data.pageSize,
      include_points: false,
    });

    const normalized = this.normalizeListResponse(resp || {});
    const mappedItems = normalized.items.map((item) => ({
      ...item,
      statusLabel: this.statusLabel(item.status),
      shotLabel: this.formatShotPrediction((item && item.shot_prediction) || null).label,
    }));
    const nextList = append ? [...this.data.list, ...mappedItems] : mappedItems;

    this.setData({
      list: nextList,
      page: normalized.page,
      totalPages: normalized.totalPages,
    });

    if (nextList.length > 0) {
      const targetId = Number(this.data.initialTaskId || 0);
      if (targetId > 0) {
        const matched = nextList.find((item) => Number(item.id) === targetId);
        if (matched) {
          this.setData({ initialTaskId: null });
          this.loadDetail(matched.id);
          return;
        }
      }
    }

    if (!this.data.selectedTaskId && nextList.length > 0) {
      this.loadDetail(nextList[0].id);
    }
  },

  async refreshAll() {
    this.setData({ loading: true, refreshing: true });
    try {
      await this.loadPage(1, false);
    } catch (error) {
      console.error('trajectory history refresh error', error);
      wx.showToast({ title: this.data.texts.refreshFailed, icon: 'none' });
    } finally {
      this.setData({ loading: false, refreshing: false });
    }
  },

  async loadMore() {
    if (this.data.page >= this.data.totalPages || this.data.loading || this.data.refreshing) return;
    try {
      await this.loadPage(this.data.page + 1, true);
    } catch (error) {
      console.error('trajectory history loadMore error', error);
    }
  },

  async loadDetail(taskId) {
    if (!taskId) return;

    this.setData({ selectedTaskId: taskId, detailLoading: true });
    try {
      const detail = await api.get(`/api/trajectory/history/${taskId}`, { include_points: true });
      const points = (((detail || {}).analysis || {}).analysis || {}).trajectory_points || [];
      const detailTracking = (((detail || {}).analysis || {}).analysis || {}).tracking || {};
      const detailMetrics = (detail || {}).metrics || {};
      const detailShotPrediction = (((detail || {}).analysis || {}).analysis || {}).shot_prediction || null;
      const detailShotView = this.formatShotPrediction(detailShotPrediction);
      const pointSamples = points.slice(0, 120).map((point) => ({
        ...point,
        timeLabel: this.formatTime(point.time),
        kindLabel: point.kind === 'predicted' ? this.data.texts.pointPredicted : this.data.texts.pointDetected,
      }));

      const videoName = detail?.artifacts?.annotated_video;
      let videoUrl = '';
      if (videoName) {
        videoUrl = await this.downloadVideoWithFallback(`${this.data.baseUrl}/uploads/trajectory/${videoName}`);
      }

      this.setData({ detail, detailTracking, detailMetrics, detailShotPrediction, detailShotView, pointSamples, videoUrl });
    } catch (error) {
      console.error('trajectory detail load error', error);
      wx.showToast({ title: this.data.texts.detailFailed, icon: 'none' });
    } finally {
      this.setData({ detailLoading: false });
    }
  },

  async onTaskTap(e) {
    const taskId = Number(e.currentTarget.dataset.id || 0);
    if (!taskId) return;
    await this.loadDetail(taskId);
  },

  onPointTap(e) {
    const seconds = Number(e.currentTarget.dataset.time || 0);
    const context = wx.createVideoContext('trajectoryHistoryVideo', this);
    if (context) {
      context.seek(seconds);
      context.play();
    }
  },

  startProcessingPoll() {
    this.stopProcessingPoll();
    this.pollTimer = setInterval(async () => {
      const hasProcessing = this.data.list.some((item) => item.status === 'processing' || item.status === 'pending');
      if (!hasProcessing) return;

      try {
        await this.loadPage(1, false);
        if (this.data.selectedTaskId) {
          await this.loadDetail(this.data.selectedTaskId);
        }
      } catch (error) {
        console.error('trajectory history poll error', error);
      }
    }, 600);
  },

  stopProcessingPoll() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  },

  downloadVideoWithFallback(url) {
    return new Promise((resolve) => {
      wx.downloadFile({
        url,
        success: (res) => {
          if (res.statusCode === 200 && res.tempFilePath) {
            resolve(res.tempFilePath);
          } else {
            resolve(url);
          }
        },
        fail: () => resolve(url),
      });
    });
  },

  formatTime(seconds) {
    const sec = Math.max(0, Math.floor(Number(seconds || 0)));
    const minute = Math.floor(sec / 60);
    const second = sec % 60;
    return `${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
  },

  formatShotPrediction(shotPrediction) {
    if (!shotPrediction || !shotPrediction.label) {
      return { label: '未知', detail: '未提供篮筐标定参数或轨迹不足' };
    }
    const confidence = `${(Number(shotPrediction.confidence || 0) * 100).toFixed(1)}%`;
    const crossing = shotPrediction.crossing_time != null
      ? `${Number(shotPrediction.crossing_time).toFixed(2)}s`
      : '无';
    if (shotPrediction.label === 'Basket') {
      return {
        label: '命中',
        detail: `置信度 ${confidence}，关键时间 ${crossing}，${shotPrediction.reason || '轨迹穿越篮筐窗口'}`,
      };
    }
    if (shotPrediction.label === 'No Basket') {
      return {
        label: '未命中',
        detail: `置信度 ${confidence}，关键时间 ${crossing}，${shotPrediction.reason || '轨迹未形成穿越'}`,
      };
    }
    return {
      label: '未知',
      detail: `置信度 ${confidence}，${shotPrediction.reason || '未形成有效判定'}`,
    };
  },
});
