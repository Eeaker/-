/**
 * video_player.js
 * 专用视频播放页面 — 视频在页面根层渲染，无 positioned 祖先
 * 通过 wx.navigateTo 传入参数：
 *   highlightUrl: HTTP 原始地址（用于 downloadFile）
 *   taskId: 任务 ID（显示用）
 *   segments: JSON 序列化的片段列表
 */
const app = getApp();

Page({
    data: {
        loading: true,
        videoUrl: '',
        taskId: '',
        segments: [],
        errorMsg: ''
    },

    onLoad(options) {
        const { highlightUrl, taskId, segments } = options;
        this.setData({
            taskId: taskId || '',
            segments: segments ? JSON.parse(decodeURIComponent(segments)) : []
        });

        if (!highlightUrl) {
            this.setData({ loading: false, errorMsg: '未找到高光视频地址' });
            return;
        }

        // 通过 downloadFile 将 HTTP 流转为本地安全路径，绕过 URL 安全检查
        wx.showLoading({ title: '加载高光流...' });
        wx.downloadFile({
            url: decodeURIComponent(highlightUrl),
            success: (res) => {
                if (res.statusCode === 200) {
                    this.setData({ videoUrl: res.tempFilePath, loading: false });
                } else {
                    this.setData({ loading: false, errorMsg: `服务器返回 ${res.statusCode}` });
                }
            },
            fail: (err) => {
                console.error('Download failed:', err);
                this.setData({ loading: false, errorMsg: '网络错误，请重试' });
            },
            complete: () => {
                wx.hideLoading();
            }
        });
    },

    // 点击片段跳转
    seekTo(e) {
        const start = e.currentTarget.dataset.start;
        const ctx = wx.createVideoContext('mainPlayer');
        if (ctx) ctx.seek(Number(start));
    },

    onVideoError(e) {
        console.error('Video error:', e.detail.errMsg);
        wx.showToast({ title: '播放失败，视频格式不兼容', icon: 'none' });
    }
});
