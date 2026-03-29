const { ref, onMounted } = Vue;

export default {
    template: `
        <div style="padding: 20px;" v-loading="loading" element-loading-text="加载神经元数据中..." element-loading-background="var(--bg-overlay-glass)">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                <h2 style="margin: 0; color: var(--text-main); font-weight: 700; letter-spacing: -0.02em;">AI 中控大屏 / Analytics</h2>
                <el-button type="primary" @click="fetchData" circle plain icon="el-icon-refresh"></el-button>
            </div>

            <div v-if="stats">
                <!-- 核心四大指标 -->
                <el-row :gutter="24">
                    <el-col :span="6">
                        <div class="premium-card stat-card">
                            <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">总检索次数</div>
                            <div class="stat-value">{{ stats.total_searches }}</div>
                        </div>
                    </el-col>
                    <el-col :span="6">
                        <div class="premium-card stat-card">
                            <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">分析视频数</div>
                            <div class="stat-value">{{ stats.total_game_analysis_tasks ?? stats.total_video_tasks }}</div>
                        </div>
                    </el-col>
                    <el-col :span="6">
                        <div class="premium-card stat-card">
                            <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500;">注册球员数</div>
                            <div class="stat-value">{{ stats.total_users }}</div>
                        </div>
                    </el-col>
                    <el-col :span="6">
                        <div class="premium-card stat-card" style="border-bottom: 3px solid var(--primary-color);">
                            <div style="color: var(--text-secondary); font-size: 14px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px;">
                                <i class="el-icon-trophy" style="color: var(--primary-color);"></i> 引擎标杆 Rank-1
                            </div>
                            <div class="stat-value">{{ stats.model_info.rank1_accuracy }}</div>
                        </div>
                    </el-col>
                </el-row>

                <!-- 图表与画廊状态 -->
                <div class="charts-container">
                    <div class="premium-card" style="padding: 24px;">
                        <h4 style="margin-top: 0; color: var(--text-main); font-size: 18px;">近7天系统检索请求趋势</h4>
                        <div id="lineChart" style="height: 320px; width: 100%;"></div>
                    </div>
                    
                    <div class="premium-card" style="padding: 24px;">
                        <h4 style="margin-top: 0; color: var(--primary-color); font-size: 18px; display: flex; align-items: center; gap: 8px;">
                            <i class="el-icon-s-platform"></i> 引擎状态矩阵
                        </h4>
                        <el-descriptions :column="1" border style="margin-top: 24px;" class="premium-descriptions">
                            <el-descriptions-item label="架构">{{ stats.model_info.architecture }}</el-descriptions-item>
                            <el-descriptions-item label="硬件">{{ stats.model_info.inference_device }}</el-descriptions-item>
                            <el-descriptions-item label="mAP">{{ stats.model_info.map_score }}%</el-descriptions-item>
                            <el-descriptions-item label="图库容量">{{ stats.model_info.gallery_size }} 张图</el-descriptions-item>
                            <el-descriptions-item label="特征维度">{{ stats.model_info.feature_dim }} D</el-descriptions-item>
                        </el-descriptions>
                    </div>
                </div>
            </div>
        </div>
    `,
    setup() {
        const loading = ref(true);
        const stats = ref({
            total_searches: '--', total_video_tasks: '--', total_game_analysis_tasks: '--', total_users: '--',
            model_info: { architecture: '--', rank1_accuracy: '--', map_score: '--', gallery_size: '--', feature_dim: '--', inference_device: '--' }
        });

        const initChart = (data) => {
            const chartDom = document.getElementById('lineChart');
            if (!chartDom) return;

            // Do not force 'dark' theme anymore, let it adapt
            const myChart = echarts.init(chartDom);
            const dates = data.map(item => item.date);
            const counts = data.map(item => item.count);

            const isDark = document.documentElement.classList.contains('dark');
            const textColor = isDark ? '#8B8E98' : '#777E90';
            const splitColor = isDark ? '#23262F' : '#E6E8EC';

            const option = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis' },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLabel: { color: textColor },
                    axisLine: { lineStyle: { color: splitColor } }
                },
                yAxis: {
                    type: 'value',
                    splitLine: { lineStyle: { color: splitColor } },
                    axisLabel: { color: textColor }
                },
                series: [
                    {
                        data: counts,
                        type: 'line',
                        smooth: true,
                        itemStyle: { color: '#E86321' },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgba(232, 99, 33, 0.4)' },
                                { offset: 1, color: 'rgba(232, 99, 33, 0.05)' }
                            ])
                        }
                    }
                ]
            };
            myChart.setOption(option);
            window.addEventListener('resize', () => myChart.resize());
        };

        const fetchData = async () => {
            loading.value = true;
            try {
                const res = await axios.get('/api/dashboard');
                const data = res.data || {};
                const mi = data.model_info || {};
                data.model_info = {
                    architecture: mi.architecture || mi.model || '--',
                    rank1_accuracy: mi.rank1_accuracy || 'N/A',
                    map_score: mi.map_score || 'N/A',
                    gallery_size: mi.gallery_size ?? '--',
                    feature_dim: mi.feature_dim ?? mi.feat_dim ?? '--',
                    inference_device: mi.inference_device || mi.device || '--'
                };
                stats.value = data;
                setTimeout(() => {
                    initChart(stats.value.daily_searches);
                }, 100);
            } catch (err) {
                ElementPlus.ElMessage.error('无法获取数据大屏面板信息');
            } finally {
                loading.value = false;
            }
        };

        onMounted(() => {
            fetchData();
        });

        return { loading, stats, fetchData };
    }
};
