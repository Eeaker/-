const api = require('../../utils/api.js');
const app = getApp();

Page({
    data: {
        activeStep: 0,
        targetImageUrl: '',
        videoUrl: '',
        loading: false,

        taskId: null,
        progress: 0,
        taskStatus: 'idle',

        resultVideoUrl: '',
        matches: [],
        baseUrl: app.globalData.baseUrl
    },

    timer: null,

    // --- 核心容错: 生命周期函数防杀干预 ---
    onShow() {
        this.setData({ baseUrl: app.globalData.baseUrl });
        const stashedTaskId = wx.getStorageSync('processing_video_taskId');

        if (stashedTaskId && this.data.activeStep === 1) {
            console.log('Wake up: Resuming polling for task:', stashedTaskId);
            // BUG FIX: Must use setData, not direct mutation
            this.setData({ taskId: stashedTaskId });
            this.pollTaskStatus();
        }
    },

    onHide() {
        // 核心容错：小程序切入后台，必须主动清掉定时器防止内存泄漏和假死
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            console.log("Suspend: Polling paused safely.");
        }
    },

    onUnload() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    },
    // ------------------------------------

    chooseTargetImage() {
        wx.chooseMedia({
            count: 1, mediaType: ['image'], sourceType: ['album', 'camera'], sizeType: ['compressed'],
            success: (res) => { this.setData({ targetImageUrl: res.tempFiles[0].tempFilePath }); }
        });
    },

    chooseVideo() {
        // 强制系统级压缩并卡死 60s 限制，突破体积红线
        wx.chooseMedia({
            count: 1, mediaType: ['video'], sourceType: ['album', 'camera'], maxDuration: 60, compressed: true,
            success: (res) => { this.setData({ videoUrl: res.tempFiles[0].tempFilePath }); }
        });
    },

    previewTargetImage() {
        if (this.data.targetImageUrl) {
            wx.previewImage({ urls: [this.data.targetImageUrl] });
        }
    },

    async submitAnalysis() {
        if (!this.data.targetImageUrl || !this.data.videoUrl) {
            wx.showToast({ title: '目标图片和视频均需上传', icon: 'none' });
            return;
        }

        this.setData({ loading: true });

        try {
            // 步骤 1: 必须先单独上传图片取得凭证
            const qResp = await api.uploadFile({
                url: '/api/video/query/upload',
                filePath: this.data.targetImageUrl,
                name: 'query'
            });

            if (!qResp || !qResp.query_filename) {
                throw new Error("检索目标图片打包失败");
            }

            // 步骤 2: 传视频并且把刚才拿到的文件名塞进 formData
            const resData = await api.uploadFile({
                url: '/api/video/analyze',
                filePath: this.data.videoUrl,
                name: 'video',
                formData: {
                    query_filename: qResp.query_filename
                }
            });

            const taskId = resData.task_id;

            // 核心容错: 防杀保存
            wx.setStorageSync('processing_video_taskId', taskId);

            this.setData({
                loading: false,
                activeStep: 1,
                taskId: taskId,
                progress: 0,
                taskStatus: '引擎调度计算序列中...'
            });

            this.pollTaskStatus();

        } catch (e) {
            console.error(e);
            this.setData({ loading: false });
        }
    },

    pollTaskStatus() {
        if (this.timer) clearInterval(this.timer);

        // 基础轮询逻辑
        this.timer = setInterval(async () => {
            try {
                const res = await api.get(`/api/video/task/${this.data.taskId}`);
                this.setData({
                    // 使用后端真实进度，防止 res.progress === 0 时错误保留上一个值
                    progress: res.progress !== undefined ? res.progress : this.data.progress,
                    taskStatus: res.status || 'running'
                });

                if (res.status === 'completed') {
                    clearInterval(this.timer);
                    this.timer = null;
                    // 清理防杀标记
                    wx.removeStorageSync('processing_video_taskId');

                    // MAP RESULTS FOR WXML
                    const backendResult = res.analysis || {};
                    const deepAnalysis = backendResult.analysis || {};
                    const segments = deepAnalysis.segments || [];

                    const formattedMatches = segments.map(m => ({
                        ...m,
                        formatted_time: m.start_str || '00:00',
                        // 兼容后端返回 avg_similarity 或 best_similarity 两种字段
                        score_percent: ((m.avg_similarity || m.best_similarity || 0) * 100).toFixed(1)
                    }));

                    // Construct the absolute URL to bypass DEMUXER_ERROR_COULD_NOT_OPEN
                    const highlightUrl = deepAnalysis.highlight_video
                        ? `${app.globalData.baseUrl}/uploads/videos/${deepAnalysis.highlight_video}`
                        : '';

                    this.setData({
                        activeStep: 2,
                        resultVideoUrl: '',
                        matches: formattedMatches
                    });

                    if (highlightUrl) {
                        wx.showLoading({ title: '加载加密高光流...' });
                        wx.downloadFile({
                            url: highlightUrl,
                            success: (dRes) => {
                                if (dRes.statusCode === 200) {
                                    this.setData({ resultVideoUrl: dRes.tempFilePath });
                                }
                            },
                            complete: () => {
                                wx.hideLoading();
                            }
                        });
                    }

                    wx.showToast({ title: '分析完成', icon: 'success' });
                } else if (res.status === 'failed') {
                    clearInterval(this.timer);
                    this.timer = null;
                    wx.removeStorageSync('processing_video_taskId');
                    wx.showModal({ title: '引擎级崩溃', content: '云端计算失败，请稍后重试', showCancel: false });
                    this.setData({ activeStep: 0 });
                }
            } catch (e) {
                console.error("Polling fault. Will retry naturally via interval.");
            }
        }, 2000);
    },

    // --- 核心容错: 底层静默失败拦截与精准的 Seek ---
    playSegment(e) {
        const startTimeStamp = e.currentTarget.dataset.start;
        const ctx = wx.createVideoContext('highlightVideo', this);

        // 直接下发原生指令
        try {
            ctx.seek(startTimeStamp);
            ctx.play();
        } catch (err) {
            // 捕获 X5 内核寻帧崩溃
            wx.showToast({ title: `快速定位系统失效，请手动拖动至 ${this.formatTimeNative(startTimeStamp)}`, icon: 'none', duration: 4000 });
        }
    },

    // 微信 video 组件自带的错误捕捉钩子
    onVideoError(e) {
        console.error('Core Video Decoder Error:', e.detail.errMsg);
        wx.showToast({ title: '系统底层解码器崩溃，可能因设备老旧导致', icon: 'none' });
    },
    // ------------------------------------------------

    async resetAnalysis() {
        // 停止轮询
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }

        // 如果有进行中的任务，通知后端取消（匹配 Web 端逻辑）
        if (this.data.taskId && this.data.activeStep === 1) {
            try {
                await api.post(`/api/video/cancel/${this.data.taskId}`);
            } catch (e) {
                console.warn('取消后台任务失败', e);
            }
        }

        wx.removeStorageSync('processing_video_taskId');
        this.setData({
            activeStep: 0,
            targetImageUrl: '',
            videoUrl: '',
            matches: [],
            taskId: null,
            progress: 0,
            taskStatus: 'idle',
            resultVideoUrl: ''
        });
    },

    formatTimeNative(seconds) {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
    }
});
