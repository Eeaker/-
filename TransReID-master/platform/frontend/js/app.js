const { createApp, ref, reactive, onMounted, onUnmounted } = Vue;
const { createRouter, createWebHashHistory, useRouter, useRoute } = VueRouter;

// 导入页面与组件 (基于 ES Modules)
import AppLayout from './components/AppLayout.js?v=20260329b';
import LoginPage from './pages/LoginPage.js?v=20260329b';
import DashboardPage from './pages/DashboardPage.js?v=20260329b';
import ImageSearchPage from './pages/ImageSearchPage.js?v=20260329b';
import GameAnalysisPage from './pages/GameAnalysisPage.js?v=20260329b';

// --- 配置路由 ---
const routes = [
    {
        path: '/login',
        name: 'Login',
        component: LoginPage,
        meta: { requiresAuth: false }
    },
    {
        path: '/',
        component: AppLayout,
        meta: { requiresAuth: true },
        children: [
            { path: '', name: 'Dashboard', component: DashboardPage },
            { path: 'game-analysis', name: 'GameAnalysis', component: GameAnalysisPage },
            { path: 'image-search', name: 'ImageSearch', component: ImageSearchPage },
            // Legacy entries are redirected to unified game analysis.
            { path: 'video-analysis', redirect: '/game-analysis' },
            { path: 'trajectory-analysis', redirect: '/game-analysis' },
        ]
    }
];

const router = createRouter({
    history: createWebHashHistory(),
    routes,
});

// --- 路由守卫：登录拦截 ---
router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('token');
    const requiresAuth = to.matched.some(record => record.meta.requiresAuth);

    if (requiresAuth && !token) {
        next('/login');
    } else if (to.path === '/login' && token) {
        next('/');
    } else {
        next();
    }
});

// --- Axios 全局拦截器：附加 JWT Token ---
axios.interceptors.request.use(config => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

axios.interceptors.response.use(
    response => response,
    error => {
        if (error.response && error.response.status === 401) {
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            router.push('/login');
            ElementPlus.ElMessage.error('登录过期，请重新登录');
        }
        return Promise.reject(error);
    }
);

// --- 启动应用 ---
const app = createApp({
    template: `<router-view></router-view>`
});

app.use(router);
app.use(ElementPlus);

// 等待路由初始化完成再挂载，避免初次加载空白屏
router.isReady().then(() => {
    app.mount('#app');
});
