App({
    onLaunch: function () {
        console.log('BallShow MiniProgram Launched');
        // Try to load token from storage
        const token = wx.getStorageSync('access_token');
        const userInfo = wx.getStorageSync('user_info');
        if (token) {
            this.globalData.token = token;
            this.globalData.isLoggedIn = true;
        }
        if (userInfo) {
            this.globalData.userInfo = userInfo;
        }
    },

    globalData: {
        userInfo: null,
        token: null,
        isLoggedIn: false,
        // Provide a way to globally access the API base URL
        baseUrl: 'http://127.0.0.1:8000'
    }
})
