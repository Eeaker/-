const api = require('../../utils/api.js');

Page({
    data: {
        stats: {},
        gpuStatus: 'Offline',
        loading: true,
        canvasRunning: false
    },

    onShow() {
        this.fetchDashboardStats();
        if (!this.data.canvasRunning) {
            this.initCanvas();
        }
    },

    async fetchDashboardStats() {
        try {
            // 从后端读取真实的特征库统计等数据
            const data = await api.get('/api/dashboard');

            this.setData({
                stats: data || {},
                gpuStatus: data.gpu_status || 'Online',
                loading: false
            });

            // Stop skeleton loader or pulldown refresh here
        } catch (err) {
            console.error('Failed to load dashboard', err);
            // fallback to safe display
            this.setData({ gpuStatus: 'Offline (API ERROR)', loading: false });
        }
    },

    onPullDownRefresh() {
        this.fetchDashboardStats().then(() => {
            wx.stopPullDownRefresh();
        });
    },

    onHide() {
        this.setData({ canvasRunning: false });
    },

    onUnload() {
        this.setData({ canvasRunning: false });
    },

    onReady() {
        this.initCanvas();
    },

    initCanvas() {
        const query = wx.createSelectorQuery();
        query.select('#clusterCanvas')
            .fields({ node: true, size: true })
            .exec((res) => {
                if (!res || !res[0] || !res[0].node) return;
                const canvas = res[0].node;
                const ctx = canvas.getContext('2d');

                const width = res[0].width || wx.getWindowInfo().windowWidth - 40;
                const height = res[0].height || 220;
                const dpr = wx.getWindowInfo().pixelRatio;

                canvas.width = width * dpr;
                canvas.height = height * dpr;
                ctx.scale(dpr, dpr);

                this.drawScatterClusters(canvas, ctx, width, height);
            });
    },

    drawScatterClusters(canvas, ctx, width, height) {
        // Create random particles representing feature dimensions
        const particles = [];
        const colors = ['#E86321', '#FF8F54', '#4CAF50', '#8B8E98'];

        for (let i = 0; i < 150; i++) {
            // Randomly form 3 clusters
            const clusterIndex = Math.floor(Math.random() * 3);
            let centerX, centerY;
            if (clusterIndex === 0) { centerX = width * 0.3; centerY = height * 0.4; }
            else if (clusterIndex === 1) { centerX = width * 0.7; centerY = height * 0.3; }
            else { centerX = width * 0.5; centerY = height * 0.7; }

            particles.push({
                x: Math.max(0, Math.min(width, centerX + (Math.random() - 0.5) * 140)),
                y: Math.max(0, Math.min(height, centerY + (Math.random() - 0.5) * 140)),
                r: Math.random() * 2.5 + 1.5,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                color: colors[Math.floor(Math.random() * colors.length)],
                alpha: Math.random() * 0.5 + 0.3
            });
        }

        const render = () => {
            if (!this.data.canvasRunning) return;

            ctx.clearRect(0, 0, width, height);

            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;

                // Keep inside bounds softly
                if (p.x < 0 || p.x > width) p.vx *= -1;
                if (p.y < 0 || p.y > height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, 2 * Math.PI);
                ctx.fillStyle = p.color;
                ctx.globalAlpha = p.alpha;
                ctx.fill();
            });

            ctx.globalAlpha = 1.0;
            canvas.requestAnimationFrame(render);
        };

        this.setData({ canvasRunning: true });
        render();
    }
});
