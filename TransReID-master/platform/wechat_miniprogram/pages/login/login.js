const api = require('../../utils/api.js');
const app = getApp();

Page({
    data: {
        mode: 'login',
        username: 'admin',
        password: 'admin',
        nickname: '',
        loading: false
    },

    onShow() {
        wx.hideHomeButton && wx.hideHomeButton();
    },

    onUsernameInput(e) {
        this.setData({ username: e.detail.value });
    },

    onPasswordInput(e) {
        this.setData({ password: e.detail.value });
    },

    onNicknameInput(e) {
        this.setData({ nickname: e.detail.value });
    },

    switchMode(e) {
        const mode = e.currentTarget.dataset.mode;
        this.setData({ mode });
    },

    handleSubmit() {
        if (this.data.mode === 'login') {
            this.handleLogin();
            return;
        }
        this.handleRegister();
    },

    async handleLogin() {
        const username = (this.data.username || '').trim();
        const password = this.data.password || '';

        if (!username || !password) {
            wx.showToast({ title: '请输入用户名和密码', icon: 'none' });
            return;
        }

        this.setData({ loading: true });
        try {
            const res = await api.post('/api/login', { username, password });

            wx.setStorageSync('access_token', res.token);
            wx.setStorageSync('user_info', res.user || {});

            app.globalData.token = res.token;
            app.globalData.userInfo = res.user || null;
            app.globalData.isLoggedIn = true;

            wx.showToast({ title: '登录成功', icon: 'success' });
            setTimeout(() => {
                wx.switchTab({ url: '/pages/dashboard/dashboard' });
            }, 500);
        } catch (err) {
            console.error('Login Error', err);
        } finally {
            this.setData({ loading: false });
        }
    },

    async handleRegister() {
        const username = (this.data.username || '').trim();
        const password = this.data.password || '';
        const nickname = (this.data.nickname || '').trim();

        if (!username || !password) {
            wx.showToast({ title: '请输入用户名和密码', icon: 'none' });
            return;
        }

        this.setData({ loading: true });
        try {
            const res = await api.post('/api/register', { username, password, nickname });

            wx.setStorageSync('access_token', res.token);
            wx.setStorageSync('user_info', res.user || {});

            app.globalData.token = res.token;
            app.globalData.userInfo = res.user || null;
            app.globalData.isLoggedIn = true;

            wx.showToast({ title: '注册并登录成功', icon: 'success' });
            setTimeout(() => {
                wx.switchTab({ url: '/pages/dashboard/dashboard' });
            }, 500);
        } catch (err) {
            console.error('Register Error', err);
        } finally {
            this.setData({ loading: false });
        }
    }
});
