// utils/api.js
const app = getApp();

const request = (options) => {
    return new Promise((resolve, reject) => {
        // 严格读取缓存，保证只要登录过就必带 Token
        const token = wx.getStorageSync('access_token') || app.globalData.token;
        const headers = {
            'Content-Type': 'application/json',
            ...(options.header || {})
        };

        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        wx.request({
            url: `${app.globalData.baseUrl}${options.url}`,
            method: options.method || 'GET',
            data: options.data,
            header: headers,
            success: (res) => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    resolve(res.data);
                } else if (res.statusCode === 401) {
                    // Token 过期或未授权，清除状态并跳回登录
                    app.globalData.token = null;
                    app.globalData.isLoggedIn = false;
                    wx.removeStorageSync('access_token');
                    wx.showToast({ title: '登录已过期，请重新登录', icon: 'none' });
                    wx.reLaunch({ url: '/pages/login/login' });
                    reject(new Error('Unauthorized'));
                } else {
                    wx.showToast({
                        title: res.data.detail || '请求失败',
                        icon: 'none'
                    });
                    reject(res.data);
                }
            },
            fail: (err) => {
                wx.showToast({ title: '网络请求失败', icon: 'none' });
                reject(err);
            }
        });
    });
};

const uploadFile = (options) => {
    return new Promise((resolve, reject) => {
        const token = wx.getStorageSync('access_token') || app.globalData.token;
        const headers = { ...(options.header || {}) };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        wx.uploadFile({
            url: `${app.globalData.baseUrl}${options.url}`,
            filePath: options.filePath,
            name: options.name,
            formData: options.formData,
            header: headers,
            success: (res) => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        const data = JSON.parse(res.data);
                        resolve(data);
                    } catch (e) {
                        resolve(res.data);
                    }
                } else {
                    reject(res);
                }
            },
            fail: reject
        });
    });
};

module.exports = {
    request,
    uploadFile,
    get: (url, data) => request({ url, data, method: 'GET' }),
    post: (url, data) => request({ url, data, method: 'POST' })
};
