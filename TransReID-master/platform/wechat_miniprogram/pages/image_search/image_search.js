const api = require('../../utils/api.js');
const app = getApp();

Page({
    data: {
        previewUrl: '',
        uploadedFilePath: '',
        topk: 10,
        loading: false,
        results: [],
        baseUrl: app.globalData.baseUrl
    },

    onShow() {
        this.setData({ baseUrl: app.globalData.baseUrl });
    },

    chooseImage() {
        // 强制使用最新的原生 chooseMedia 规避大体积和压缩错位风险
        wx.chooseMedia({
            count: 1,
            mediaType: ['image'],
            sourceType: ['album', 'camera'],
            sizeType: ['compressed'], // 使用原生压缩减小网络负担
            success: (res) => {
                const tempFilePath = res.tempFiles[0].tempFilePath;
                this.setData({
                    previewUrl: tempFilePath,
                    uploadedFilePath: tempFilePath,
                    results: [] // 选新图清空旧结果
                });
            }
        });
    },

    // 上传框点击处理：无图时打开选图，有图时预览
    onBoxTap() {
        if (!this.data.previewUrl) {
            this.chooseImage();
        } else {
            this.previewInputImage();
        }
    },

    previewInputImage() {
        if (this.data.previewUrl) {
            wx.previewImage({ urls: [this.data.previewUrl] });
        }
    },

    onTopkChange(e) {
        this.setData({ topk: e.detail.value });
    },

    async submitSearch() {
        if (!this.data.uploadedFilePath) {
            wx.showToast({ title: '请先选取待检索图像', icon: 'none' });
            return;
        }

        this.setData({ loading: true, results: [] });

        try {
            const resp = await api.uploadFile({
                url: '/api/reid/search',
                filePath: this.data.uploadedFilePath,
                name: 'file', // FastAPI endpoint expects 'file' form-data
                formData: {
                    topk: this.data.topk.toString()
                }
            });

            if (resp && resp.results) {
                // Pre-calculate percentages for template parsing brevity
                const formattedResults = resp.results.map(r => ({
                    ...r,
                    score_percent: (r.score * 100).toFixed(1)
                }));

                this.setData({ results: formattedResults });
                wx.showToast({ title: '检索完成', icon: 'success' });
            }
        } catch (err) {
            console.error('Image Search Error', err);
            // err object format can vary from wx API, api.js should have toasted generic
        } finally {
            this.setData({ loading: false });
        }
    },

    previewResult(e) {
        const index = e.currentTarget.dataset.index;
        const current = this.data.baseUrl + this.data.results[index].path;
        const urls = this.data.results.map(r => this.data.baseUrl + r.path);

        // 完美解决 Web 端 Zoom 闪屏痛点的杀手锏：系统级全屏 API
        wx.previewImage({
            current: current,
            urls: urls,
            showmenu: true // 允许长按保存提取等
        });
    }
});
