const app = getApp();
const api = require('../../utils/api.js');

Page({
  data: {
    userInfo: null,
    displayName: '\u672a\u767b\u5f55\u7528\u6237',
    displayUsername: '--',
    avatarLetter: 'U',
    highlightCount: 0,
    trajectoryCount: 0,
    loading: false,
    texts: {
      userLabel: '\u7528\u6237\u540d\uff1a',
      logout: '\u9000\u51fa\u767b\u5f55',
      sectionTitle: '\u5386\u53f2\u8bb0\u5f55\u5165\u53e3',
      sectionSub: '\u5206\u522b\u8fdb\u5165\u9ad8\u5149\u96c6\u9526\u4e0e\u8f68\u8ff9\u9884\u6d4b\u5386\u53f2',
      highlightTitle: '\u9ad8\u5149\u96c6\u9526\u5386\u53f2',
      highlightDesc: '\u67e5\u770b\u89c6\u9891\u5206\u6790\u4efb\u52a1\u3001\u9ad8\u5149\u7247\u6bb5\u4e0e\u56de\u653e',
      trajectoryTitle: '\u8f68\u8ff9\u9884\u6d4b\u5386\u53f2',
      trajectoryDesc: '\u67e5\u770b\u68c0\u6d4b\u70b9\u3001\u8986\u76d6\u7387\u53ca\u8f68\u8ff9\u8be6\u60c5\u56de\u653e',
      open: '\u8fdb\u5165',
      loadFailed: '\u5386\u53f2\u7edf\u8ba1\u52a0\u8f7d\u5931\u8d25'
    }
  },

  onShow() {
    const userInfo = wx.getStorageSync('user_info') || app.globalData.userInfo || null;
    const displayName = (userInfo && (userInfo.nickname || userInfo.username)) || this.data.displayName;
    const displayUsername = (userInfo && userInfo.username) || '--';
    const avatarLetter = displayName ? displayName.slice(0, 1) : 'U';
    this.setData({ userInfo, displayName, displayUsername, avatarLetter });
    this.loadCounts();
  },

  async loadCounts() {
    this.setData({ loading: true });
    try {
      const [videoResp, trajectoryResp] = await Promise.all([
        api.get('/api/video/history').catch(() => []),
        api.get('/api/trajectory/history', { page: 1, page_size: 1, include_points: false }).catch(() => ({ items: [], pagination: { total: 0 } })),
      ]);

      const highlightCount = Array.isArray(videoResp)
        ? videoResp.length
        : Array.isArray(videoResp?.items)
          ? videoResp.items.length
          : 0;
      const trajectoryCount = Array.isArray(trajectoryResp)
        ? trajectoryResp.length
        : Number(trajectoryResp?.pagination?.total || (Array.isArray(trajectoryResp?.items) ? trajectoryResp.items.length : 0));

      this.setData({ highlightCount, trajectoryCount, loading: false });
    } catch (e) {
      this.setData({ loading: false });
      wx.showToast({ title: this.data.texts.loadFailed, icon: 'none' });
    }
  },

  goHighlightHistory() {
    wx.navigateTo({ url: '/pages/highlight_history/highlight_history' });
  },

  goTrajectoryHistory() {
    wx.navigateTo({ url: '/pages/trajectory_history/trajectory_history' });
  },

  logout() {
    wx.removeStorageSync('access_token');
    wx.removeStorageSync('user_info');
    app.globalData.token = null;
    app.globalData.userInfo = null;
    app.globalData.isLoggedIn = false;
    wx.reLaunch({ url: '/pages/login/login' });
  },
});

