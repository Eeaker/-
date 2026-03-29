const api = require('../../utils/api.js');
const app = getApp();

const NAV_ITEM_WIDTH = 160;
const NAV_ITEM_GAP = 24;

Page({
  data: {
    texts: {
      headerTitle: '功能中心',
      headerSub: '拖动上方功能栏，将要使用的功能滑到中间即可切换',

      imageTitle: '图片检索',
      imageDesc: '上传一张查询图片，执行全库检索匹配',
      chooseImage: '选择图片',
      imageSelected: '已选择图片',
      imageNotSelected: '未选择图片',
      topkPrefix: 'Top-K: ',
      startSearch: '开始检索',

      highlightTitle: '高光集锦',
      highlightDesc: '上传目标图和比赛视频，系统自动生成高光视频并支持时间跳转',
      chooseTarget: '选择目标图',
      chooseVideo: '选择比赛视频',
      targetLabel: '目标图',
      videoLabel: '比赛视频',
      targetEmpty: '未选择目标图',
      videoEmpty: '未选择比赛视频',
      startAnalyze: '开始分析',
      resultTitle: '高光结果',
      resultPreparing: '结果视频准备中...',
      segmentPrefix: '高光位置 ',
      durationPrefix: '时长 ',
      scorePrefix: '置信度 ',
      framePrefix: '帧数 ',

      trajectoryTitle: '投篮轨迹预测',
      trajectoryDesc: '上传投篮视频，系统使用 YOLO + HSV + 卡尔曼滤波进行轨迹检测并绘制到视频上',
      chooseShotVideo: '选择投篮视频',
      shotVideoSelected: '已选择投篮视频',
      shotVideoEmpty: '未选择投篮视频',
      startTrajectory: '开始轨迹分析',
      trajectoryResult: '轨迹预测结果',
      shotPredictionTitle: '投篮判定',
      shotUnknown: '未知',
      shotHit: '命中',
      shotMiss: '未命中'
    },
    navItems: [
      { key: 'image', label: '图片检索' },
      { key: 'highlight', label: '高光集锦' },
      { key: 'trajectory', label: '轨迹预测' }
    ],
    selectedIndex: 0,
    navScrollLeft: 0,
    navSidePadding: 32,
    navItemWidth: NAV_ITEM_WIDTH,
    navItemGap: NAV_ITEM_GAP,

    baseUrl: app.globalData.baseUrl,

    imageQueryPath: '',
    imageTopk: 20,
    imageLoading: false,
    imageResults: [],

    videoTargetPath: '',
    videoPath: '',
    videoLoading: false,
    videoTaskId: null,
    videoProgress: 0,
    videoStatus: 'idle',
    videoResultUrl: '',
    videoSegments: [],
    activeSegmentIndex: -1,

    trajectoryVideoPath: '',
    trajectoryLoading: false,
    trajectoryTaskId: null,
    trajectoryProgress: 0,
    trajectoryStatus: 'idle',
    trajectoryResultUrl: '',
    trajectoryTracking: null,
    trajectoryPoints: [],
    trajectoryShotPrediction: null,
    trajectoryShotView: {
      label: '未知',
      detail: '未提供篮筐标定参数或轨迹不足'
    },
    activeTrajectoryPointIndex: -1,
    trajectoryLiveState: {
      processed_frame: 0,
      last_detected_frame: -1,
      last_emitted_frame: -1,
      lag_frames: 0,
      phase: 'pre_shot',
      causal_mode: true
    }
  },

  videoPollTimer: null,
  trajectoryPollTimer: null,
  navAutoScrollTimer: null,
  lastNavScrollLeft: 0,
  isNavAutoScrolling: false,
  isUserDraggingNav: false,
  ignoreNextTouchEnd: false,
  navTouchStartX: 0,
  navTouchMoved: false,

  onLoad() {
    this.setData({ baseUrl: app.globalData.baseUrl });
    this.recalcNavSidePadding();
  },

  onReady() {
    this.recalcNavSidePadding();
    this.centerToIndex(0, false);
  },

  onShow() {
    this.recalcNavSidePadding();
    this.centerToIndex(this.data.selectedIndex, false);
  },

  onTabItemTap() {
    this.recalcNavSidePadding();
    this.centerToIndex(this.data.selectedIndex, false);
  },

  onHide() {
    this.clearVideoPollTimer();
    this.clearTrajectoryPollTimer();
    this.clearNavAutoScrollTimer();
  },

  onUnload() {
    this.clearVideoPollTimer();
    this.clearTrajectoryPollTimer();
    this.clearNavAutoScrollTimer();
  },

  resetFeatureCenterToImage() {
    this.recalcNavSidePadding();
    this.centerToIndex(0, true);
  },

  recalcNavSidePadding() {
    let windowWidth = 375;
    try {
      windowWidth = wx.getWindowInfo().windowWidth;
    } catch (e) {
      windowWidth = wx.getSystemInfoSync().windowWidth;
    }
    const pageHorizontalPaddingPx = (windowWidth * 30 / 750) * 2;
    const navViewportWidth = Math.max(0, windowWidth - pageHorizontalPaddingPx);
    const navSidePadding = Math.max(0, (navViewportWidth - this.data.navItemWidth) / 2);
    this.setData({ navSidePadding });
  },

  clearVideoPollTimer() {
    if (this.videoPollTimer) {
      clearInterval(this.videoPollTimer);
      this.videoPollTimer = null;
    }
  },

  clearTrajectoryPollTimer() {
    if (this.trajectoryPollTimer) {
      clearInterval(this.trajectoryPollTimer);
      this.trajectoryPollTimer = null;
    }
  },

  clearNavAutoScrollTimer() {
    if (this.navAutoScrollTimer) {
      clearTimeout(this.navAutoScrollTimer);
      this.navAutoScrollTimer = null;
    }
    this.isNavAutoScrolling = false;
  },

  onNavTouchStart(e) {
    const touch = (e.touches && e.touches[0]) || {};
    this.navTouchStartX = Number(touch.pageX || 0);
    this.navTouchMoved = false;
    this.isUserDraggingNav = false;
    this.clearNavAutoScrollTimer();
  },

  onNavTouchMove(e) {
    const touch = (e.touches && e.touches[0]) || {};
    const currentX = Number(touch.pageX || 0);
    if (Math.abs(currentX - this.navTouchStartX) > 6) {
      this.navTouchMoved = true;
      this.isUserDraggingNav = true;
    }
  },

  onNavTouchEnd() {
    if (this.ignoreNextTouchEnd) {
      this.ignoreNextTouchEnd = false;
      this.isUserDraggingNav = false;
      this.navTouchMoved = false;
      return;
    }
    if (!this.navTouchMoved) {
      this.isUserDraggingNav = false;
      return;
    }
    if (!this.isUserDraggingNav) return;
    this.isUserDraggingNav = false;
    this.navTouchMoved = false;
    if (this.isNavAutoScrolling) return;
    this.snapNavToNearest(true);
  },

  onNavTouchCancel() {
    this.onNavTouchEnd();
  },

  onNavScroll(e) {
    const span = this.data.navItemWidth + this.data.navItemGap;
    const offset = e.detail.scrollLeft || 0;
    this.lastNavScrollLeft = offset;

    // Ignore transient scroll events caused by programmatic centering.
    if (this.isNavAutoScrolling) {
      return;
    }

    // Ignore callbacks that are not from user's drag gesture.
    if (!this.isUserDraggingNav) {
      return;
    }

    const maxIndex = this.data.navItems.length - 1;
    const index = Math.max(0, Math.min(maxIndex, Math.round(offset / span)));

    if (index !== this.data.selectedIndex) {
      this.setData({ selectedIndex: index });
    }

  },

  onNavTap(e) {
    const index = Number(e.currentTarget.dataset.index || 0);
    this.ignoreNextTouchEnd = true;
    this.isUserDraggingNav = false;
    this.navTouchMoved = false;
    this.centerToIndex(index, true);
  },

  snapNavToNearest(animated = true) {
    const span = this.data.navItemWidth + this.data.navItemGap;
    const maxIndex = this.data.navItems.length - 1;
    const nearest = Math.max(0, Math.min(maxIndex, Math.round(this.lastNavScrollLeft / span)));
    this.centerToIndex(nearest, animated);
  },

  centerToIndex(index, animated = true) {
    const maxIndex = this.data.navItems.length - 1;
    const fixed = Math.max(0, Math.min(maxIndex, index));
    const span = this.data.navItemWidth + this.data.navItemGap;
    const scrollLeft = fixed * span;

    this.lastNavScrollLeft = scrollLeft;
    this.clearNavAutoScrollTimer();
    this.isNavAutoScrolling = animated;
    if (animated) {
      this.setData({ selectedIndex: fixed, navScrollLeft: scrollLeft });
      this.navAutoScrollTimer = setTimeout(() => {
        this.isNavAutoScrolling = false;
        this.navAutoScrollTimer = null;
      }, 260);
    } else {
      this.isNavAutoScrolling = false;
      this.setData({ selectedIndex: fixed, navScrollLeft: scrollLeft });
    }
  },

  chooseImageQuery() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: (res) => {
        this.setData({ imageQueryPath: res.tempFiles[0].tempFilePath });
      }
    });
  },

  previewImageQuery() {
    if (!this.data.imageQueryPath) return;
    wx.previewImage({ current: this.data.imageQueryPath, urls: [this.data.imageQueryPath] });
  },

  onTopkChange(e) {
    this.setData({ imageTopk: Number(e.detail.value || 20) });
  },

  async submitImageSearch() {
    if (!this.data.imageQueryPath) {
      wx.showToast({ title: '请先选择检索图片', icon: 'none' });
      return;
    }

    this.setData({ imageLoading: true, imageResults: [] });
    try {
      const response = await api.uploadFile({
        url: '/api/reid/search',
        filePath: this.data.imageQueryPath,
        name: 'file',
        formData: { topk: this.data.imageTopk }
      });
      const results = (response.results || []).map((item) => ({
        ...item,
        fullUrl: `${this.data.baseUrl}${item.path}`,
        scorePercent: ((item.score || 0) * 100).toFixed(1)
      }));
      this.setData({ imageResults: results });
    } catch (error) {
      console.error('submitImageSearch error', error);
      wx.showToast({ title: '图片检索失败，请重试', icon: 'none' });
    } finally {
      this.setData({ imageLoading: false });
    }
  },

  previewSearchResult(e) {
    const index = Number(e.currentTarget.dataset.index || 0);
    const urls = this.data.imageResults.map((item) => item.fullUrl).filter(Boolean);
    if (!urls[index]) return;
    wx.previewImage({ current: urls[index], urls });
  },

  chooseVideoTarget() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: (res) => {
        this.setData({ videoTargetPath: res.tempFiles[0].tempFilePath });
      }
    });
  },

  previewVideoTarget() {
    if (!this.data.videoTargetPath) return;
    wx.previewImage({ current: this.data.videoTargetPath, urls: [this.data.videoTargetPath] });
  },

  chooseVideoFile() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album', 'camera'],
      maxDuration: 300,
      compressed: true,
      success: (res) => {
        this.setData({ videoPath: res.tempFiles[0].tempFilePath });
      }
    });
  },

  async submitVideoTask() {
    if (!this.data.videoTargetPath || !this.data.videoPath) {
      wx.showToast({ title: '请先选择目标图和比赛视频', icon: 'none' });
      return;
    }

    this.clearVideoPollTimer();
    this.setData({
      videoLoading: true,
      videoProgress: 0,
      videoStatus: 'processing',
      videoTaskId: null,
      videoResultUrl: '',
      videoSegments: [],
      activeSegmentIndex: -1
    });

    try {
      const queryResp = await api.uploadFile({
        url: '/api/video/query/upload',
        filePath: this.data.videoTargetPath,
        name: 'query'
      });

      const videoResp = await api.uploadFile({
        url: '/api/video/analyze',
        filePath: this.data.videoPath,
        name: 'video',
        formData: { query_filename: queryResp.query_filename }
      });

      this.setData({ videoTaskId: videoResp.task_id, videoLoading: false });
      this.startVideoPolling();
    } catch (error) {
      console.error('submitVideoTask error', error);
      this.setData({ videoLoading: false, videoStatus: 'failed' });
      wx.showToast({ title: '提交失败，请重试', icon: 'none' });
    }
  },

  startVideoPolling() {
    this.clearVideoPollTimer();
    this.videoPollTimer = setInterval(async () => {
      const taskId = this.data.videoTaskId;
      if (!taskId) return;
      try {
        const response = await api.get(`/api/video/task/${taskId}`);
        const patch = { videoStatus: response.status || 'processing' };
        if (typeof response.progress === 'number') {
          patch.videoProgress = Math.max(0, Math.min(100, response.progress));
        }

        if (response.status === 'completed') {
          this.clearVideoPollTimer();
          patch.videoProgress = 100;
          const detail = (response.analysis || {}).analysis || {};
          const rawSegments = detail.segments || [];

          let cursor = 0;
          patch.videoSegments = rawSegments.map((seg, index) => {
            const duration = Number(seg.duration || (seg.end_time - seg.start_time) || 0);
            const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
            const highlightStart = Number(cursor.toFixed(2));
            cursor += safeDuration;
            const sourceStart = Number(seg.start_time || 0);
            const score = Number(seg.avg_similarity || seg.best_similarity || 0);
            return {
              id: index + 1,
              sourceStartLabel: seg.start_str || this.formatTime(sourceStart),
              highlightStart,
              highlightStartLabel: this.formatTime(highlightStart),
              duration: safeDuration.toFixed(1),
              frameCount: seg.frame_count || 0,
              scorePercent: (score * 100).toFixed(1)
            };
          });

          if (detail.highlight_video) {
            const remoteUrl = `${this.data.baseUrl}/uploads/videos/${detail.highlight_video}`;
            patch.videoResultUrl = await this.downloadVideoWithFallback(remoteUrl);
          }
        }

        if (response.status === 'failed' || response.status === 'cancelled') {
          this.clearVideoPollTimer();
        }

        this.setData(patch);
      } catch (error) {
        console.error('startVideoPolling error', error);
      }
    }, 2000);
  },

  chooseTrajectoryVideo() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album', 'camera'],
      maxDuration: 300,
      compressed: true,
      success: (res) => {
        this.setData({ trajectoryVideoPath: res.tempFiles[0].tempFilePath });
      }
    });
  },

  async submitTrajectoryTask() {
    if (!this.data.trajectoryVideoPath) {
      wx.showToast({ title: '请先选择投篮视频', icon: 'none' });
      return;
    }

    this.clearTrajectoryPollTimer();
    this.setData({
      trajectoryLoading: true,
      trajectoryTaskId: null,
      trajectoryProgress: 0,
      trajectoryStatus: 'processing',
      trajectoryResultUrl: '',
      trajectoryTracking: null,
      trajectoryPoints: [],
      trajectoryShotPrediction: null,
      trajectoryShotView: {
        label: this.data.texts.shotUnknown,
        detail: '未提供篮筐标定参数或轨迹不足'
      },
      activeTrajectoryPointIndex: -1,
      trajectoryLiveState: {
        processed_frame: 0,
        last_detected_frame: -1,
        last_emitted_frame: -1,
        lag_frames: 0,
        phase: 'pre_shot',
        causal_mode: true
      }
    });

    try {
      const response = await api.uploadFile({
        url: '/api/trajectory/analyze',
        filePath: this.data.trajectoryVideoPath,
        name: 'video'
      });
      this.setData({ trajectoryTaskId: response.task_id, trajectoryLoading: false });
      this.startTrajectoryPolling();
    } catch (error) {
      console.error('submitTrajectoryTask error', error);
      this.setData({ trajectoryLoading: false, trajectoryStatus: 'failed' });
      wx.showToast({ title: '提交失败，请重试', icon: 'none' });
    }
  },

  startTrajectoryPolling() {
    this.clearTrajectoryPollTimer();
    this.trajectoryPollTimer = setInterval(async () => {
      const taskId = this.data.trajectoryTaskId;
      if (!taskId) return;
      try {
        const response = await api.get(`/api/trajectory/task/${taskId}`);
        const patch = { trajectoryStatus: response.status || 'processing' };
        if (typeof response.progress === 'number') {
          patch.trajectoryProgress = Math.max(0, Math.min(100, response.progress));
        }
        if (response.live_state) {
          patch.trajectoryLiveState = {
            ...(this.data.trajectoryLiveState || {}),
            ...response.live_state
          };
        }

        if (response.status === 'completed') {
          this.clearTrajectoryPollTimer();
          patch.trajectoryProgress = 100;
          const result = response.analysis || {};
          patch.trajectoryTracking = result.analysis?.tracking || null;
          patch.trajectoryPoints = this.normalizeTrajectoryPoints(result.analysis?.trajectory_points || []);
          patch.trajectoryShotPrediction = result.analysis?.shot_prediction || null;
          patch.trajectoryShotView = this.formatShotPrediction(result.analysis?.shot_prediction || null);
          patch.activeTrajectoryPointIndex = -1;
          patch.trajectoryLiveState = {
            ...(patch.trajectoryLiveState || this.data.trajectoryLiveState || {}),
            ...(result.analysis?.live_summary || {})
          };

          const annotatedVideo = result.artifacts?.annotated_video;
          if (annotatedVideo) {
            const remoteUrl = `${this.data.baseUrl}/uploads/trajectory/${annotatedVideo}`;
            patch.trajectoryResultUrl = await this.downloadVideoWithFallback(remoteUrl);
          }
        }

        if (response.status === 'failed' || response.status === 'cancelled') {
          this.clearTrajectoryPollTimer();
        }

        this.setData(patch);
      } catch (error) {
        console.error('startTrajectoryPolling error', error);
      }
    }, 220);
  },

  downloadVideoWithFallback(url) {
    return new Promise((resolve) => {
      wx.showLoading({ title: '加载视频中' });
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
        complete: () => wx.hideLoading()
      });
    });
  },

  onSegmentTap(e) {
    const start = Number(e.currentTarget.dataset.start || 0);
    const index = Number(e.currentTarget.dataset.index || -1);
    const context = wx.createVideoContext('highlightVideo', this);
    if (context) {
      context.seek(start);
      context.play();
    }
    this.setData({ activeSegmentIndex: index });
  },

  onResultVideoError() {
    wx.showToast({ title: '高光视频播放失败，请重试', icon: 'none' });
  },

  onTrajectoryVideoError() {
    wx.showToast({ title: '轨迹视频播放失败，请重试', icon: 'none' });
  },

  normalizeTrajectoryPoints(rawPoints) {
    const points = Array.isArray(rawPoints) ? rawPoints : [];
    const normalized = points.map((p) => ({
      frame: Number(p.frame || 0),
      time: Number(p.time || 0),
      timeLabel: this.formatTime(Number(p.time || 0)),
      x: Number(p.x || 0).toFixed(1),
      y: Number(p.y || 0).toFixed(1),
      kind: p.kind === 'predicted' ? '预测' : '检测',
      phase: p.phase || 'pre_shot',
      phaseLabel: this.formatTrajectoryPhase(p.phase || 'pre_shot'),
      lagFrames: Number(p.lag_frames || 0),
      speed: Number(p.speed_px_s || 0).toFixed(1),
      confidence: Number((p.confidence || 0) * 100).toFixed(1)
    }));

    if (normalized.length <= 120) {
      return normalized;
    }

    const step = Math.max(1, Math.floor(normalized.length / 120));
    const sampled = [];
    for (let i = 0; i < normalized.length; i += step) {
      sampled.push(normalized[i]);
    }
    const last = normalized[normalized.length - 1];
    if (!sampled.length || sampled[sampled.length - 1].frame !== last.frame) {
      sampled.push(last);
    }
    return sampled.slice(0, 120);
  },

  onTrajectoryPointTap(e) {
    const seconds = Number(e.currentTarget.dataset.time || 0);
    const index = Number(e.currentTarget.dataset.index || -1);
    const context = wx.createVideoContext('trajectoryVideoResult', this);
    if (context) {
      context.seek(seconds);
      context.play();
    }
    this.setData({ activeTrajectoryPointIndex: index });
  },

  formatTime(seconds) {
    const sec = Math.max(0, Math.floor(Number(seconds || 0)));
    const minute = Math.floor(sec / 60);
    const second = sec % 60;
    return `${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
  },

  formatTrajectoryPhase(phase) {
    if (phase === 'flight') return '飞行段';
    if (phase === 'post_shot') return '出手后';
    return '出手前';
  },

  formatShotPrediction(shotPrediction) {
    if (!shotPrediction || !shotPrediction.label) {
      return {
        label: this.data.texts.shotUnknown,
        detail: '未提供篮筐标定参数或轨迹不足',
      };
    }
    const confidence = `${(Number(shotPrediction.confidence || 0) * 100).toFixed(1)}%`;
    const crossing = shotPrediction.crossing_time != null
      ? `${Number(shotPrediction.crossing_time).toFixed(2)}s`
      : '无';
    if (shotPrediction.label === 'Basket') {
      return {
        label: this.data.texts.shotHit,
        detail: `置信度 ${confidence}，关键时间 ${crossing}，${shotPrediction.reason || '轨迹穿越篮筐窗口'}`,
      };
    }
    if (shotPrediction.label === 'No Basket') {
      return {
        label: this.data.texts.shotMiss,
        detail: `置信度 ${confidence}，关键时间 ${crossing}，${shotPrediction.reason || '轨迹未形成穿越'}`,
      };
    }
    return {
      label: this.data.texts.shotUnknown,
      detail: `置信度 ${confidence}，${shotPrediction.reason || '未形成有效判定'}`,
    };
  }
});