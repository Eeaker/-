const api = require('../../utils/api.js');
const app = getApp();

Page({
  data: {
    loading: true,
    list: [],
    texts: {
      title: '\u9ad8\u5149\u96c6\u9526\u5386\u53f2',
      subtitle: '\u67e5\u770b\u6bcf\u6b21\u89c6\u9891\u5206\u6790\u7684\u9ad8\u5149\u751f\u6210\u7ed3\u679c',
      loading: '\u52a0\u8f7d\u4e2d...',
      empty: '\u6682\u65e0\u9ad8\u5149\u5386\u53f2\u8bb0\u5f55',
      taskPrefix: '\u4efb\u52a1 #',
      frameLabel: '\u5e27\uff1a',
      segmentLabel: '\u7247\u6bb5\uff1a',
      completed: '\u5df2\u5b8c\u6210',
      processing: '\u5904\u7406\u4e2d',
      play: '\u64ad\u653e\u89c2\u770b\u9ad8\u5149\u89c6\u9891',
      loadFailed: '\u9ad8\u5149\u5386\u53f2\u52a0\u8f7d\u5931\u8d25'
    }
  },

  onShow() {
    this.loadHistory();
  },

  onPullDownRefresh() {
    this.loadHistory().finally(() => wx.stopPullDownRefresh());
  },

  statusLabel(status) {
    if (status === 'completed') return this.data.texts.completed;
    if (status === 'processing') return this.data.texts.processing;
    return status || 'pending';
  },

  parseVideoHistory(resp) {
    const list = Array.isArray(resp) ? resp : (resp?.items || []);
    return list.map((item) => {
      const analysis = item.analysis || {};
      const inner = analysis.analysis || {};
      const highlightFile = inner.highlight_video || null;
      const segments = inner.segments || [];
      return {
        id: item.id,
        status: item.status || 'pending',
        statusLabel: this.statusLabel(item.status || 'pending'),
        date: item.created_at ? String(item.created_at).replace('T', ' ').slice(0, 19) : '--',
        totalFrames: item.total_frames || 0,
        matchedSegments: item.matched_segments || 0,
        highlightFile,
        hasHighlight: !!highlightFile,
        segments: segments.map((seg) => ({
          time: seg.start_time,
          formatted_time: seg.start_str || '00:00',
          score_percent: Math.round((seg.avg_similarity || seg.best_similarity || 0) * 100),
          highlight_time: seg.highlight_start,
          duration: seg.duration,
        })),
      };
    });
  },

  async loadHistory() {
    this.setData({ loading: true });
    try {
      const resp = await api.get('/api/video/history').catch(() => []);
      this.setData({ list: this.parseVideoHistory(resp), loading: false });
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: this.data.texts.loadFailed, icon: 'none' });
    }
  },

  openHighlight(e) {
    const item = e.currentTarget.dataset.item;
    if (!item || !item.highlightFile) return;

    const highlightUrl = `${app.globalData.baseUrl}/uploads/videos/${item.highlightFile}`;
    const segments = encodeURIComponent(JSON.stringify(item.segments || []));
    wx.navigateTo({
      url: `/pages/video_player/video_player?highlightUrl=${encodeURIComponent(highlightUrl)}&taskId=${item.id}&segments=${segments}`,
    });
  },
});

