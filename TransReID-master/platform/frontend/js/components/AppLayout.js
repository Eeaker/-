const { ref } = Vue;
const { useRouter, useRoute } = VueRouter;

export default {
    template: `
        <el-container class="layout-container">
            <el-aside width="260px" class="premium-aside">
                <div style="padding: 30px 20px 20px; display: flex; justify-content: center;">
                    <a href="#" class="logo">
                        <img src="https://img.icons8.com/color/48/000000/basketball.png" alt="logo" style="width: 36px;">
                        <span>BALLSHOW</span>
                    </a>
                </div>

                <el-menu
                    :default-active="activeMenu"
                    class="el-menu-vertical"
                    background-color="transparent"
                    text-color="var(--text-main)"
                    active-text-color="var(--primary-color)"
                    style="border-right: none;"
                    router
                >
                    <div class="menu-group-title">数据总览</div>
                    <el-menu-item index="/">
                        <el-icon><i class="el-icon-data-board"></i></el-icon>
                        <template #title>AI 数据总览</template>
                    </el-menu-item>

                    <div class="menu-group-title">核心引擎</div>
                    <el-menu-item index="/game-analysis">
                        <el-icon><i class="el-icon-monitor"></i></el-icon>
                        <template #title>篮球智能分析</template>
                    </el-menu-item>
                    <el-menu-item index="/image-search">
                        <el-icon><i class="el-icon-picture"></i></el-icon>
                        <template #title>全库球衣识图</template>
                    </el-menu-item>
                </el-menu>
            </el-aside>

            <el-container>
                <el-header height="64px" class="premium-header">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <el-tag type="warning" effect="dark" round style="font-family: 'Outfit', sans-serif; font-weight: 600;">Single ReID</el-tag>
                        <el-tag type="info" effect="plain" round style="font-family: 'Outfit', sans-serif; font-weight: 600; border-color: var(--border-color); color: var(--text-main);">YOLO Tracking</el-tag>
                    </div>
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <el-switch
                            v-model="isDark"
                            inline-prompt
                            active-icon="el-icon-moon"
                            inactive-icon="el-icon-sunny"
                            @change="toggleTheme"
                            style="--el-switch-on-color: #2C2C2E; --el-switch-off-color: #E5E5EA;"
                        />
                        <span style="font-size: 15px; font-weight: 600; color: var(--text-main);">
                            当前用户 <span style="color: var(--primary-color);">{{ username }}</span>
                        </span>
                        <el-button color="var(--border-color)" :dark="isDark" plain round @click="logout" size="small" style="color: var(--text-main); font-weight: 600;">退出登录</el-button>
                    </div>
                </el-header>

                <el-main style="background: transparent;">
                    <router-view v-slot="{ Component }">
                        <transition name="el-fade-in-linear" mode="out-in">
                            <component :is="Component" />
                        </transition>
                    </router-view>
                </el-main>
            </el-container>
        </el-container>
    `,
    setup() {
        const router = useRouter();
        const route = useRoute();
        const activeMenu = ref(route.path);

        const isDark = ref(document.documentElement.classList.contains("dark"));
        const toggleTheme = (val) => {
            if (val) {
                document.documentElement.classList.add("dark");
            } else {
                document.documentElement.classList.remove("dark");
            }
        };

        const userStr = localStorage.getItem("user");
        const username = ref(userStr ? JSON.parse(userStr).nickname : "管理员");

        const logout = () => {
            localStorage.removeItem("token");
            localStorage.removeItem("user");
            router.push("/login");
        };

        return { activeMenu, username, logout, isDark, toggleTheme };
    },
};


