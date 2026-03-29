const { ref, reactive } = Vue;
const { useRouter } = VueRouter;

export default {
    template: `
        <div class="login-container">
            <el-card class="login-card">
                <template #header>
                    <div style="text-align: center; color: var(--primary-color);">
                        <h2 style="margin: 0; font-weight: 900; letter-spacing: 2px;">BALLSHOW PRO</h2>
                        <span style="font-size: 12px; color: #888;">TransReID Engine</span>
                    </div>
                </template>

                <el-form :model="loginForm" label-position="top">
                    <el-form-item label="用户名">
                        <el-input v-model="loginForm.username" placeholder="admin"></el-input>
                    </el-form-item>
                    <el-form-item label="密码">
                        <el-input v-model="loginForm.password" type="password" show-password placeholder="admin"></el-input>
                    </el-form-item>
                    <el-button type="primary" style="width: 100%; margin-top: 10px;" @click="handleLogin" :loading="loading">
                        登录
                    </el-button>
                </el-form>

                <div style="margin-top: 14px; color: #8b8e98; font-size: 12px; text-align: center;">
                    默认管理员账号：admin / admin
                </div>
            </el-card>
        </div>
    `,
    setup() {
        const router = useRouter();
        const loading = ref(false);
        const loginForm = reactive({ username: "admin", password: "admin" });

        const handleLogin = async () => {
            if (!loginForm.username || !loginForm.password) {
                ElementPlus.ElMessage.warning("请输入用户名和密码");
                return;
            }
            loading.value = true;
            try {
                const res = await axios.post("/api/login", loginForm);
                localStorage.setItem("token", res.data.token);
                localStorage.setItem("user", JSON.stringify(res.data.user));
                ElementPlus.ElMessage.success("登录成功");
                router.push("/");
            } catch (err) {
                ElementPlus.ElMessage.error(err.response?.data?.detail || "登录失败");
            } finally {
                loading.value = false;
            }
        };

        return { loginForm, loading, handleLogin };
    },
};
