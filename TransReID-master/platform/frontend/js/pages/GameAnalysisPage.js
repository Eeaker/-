const { ref, computed, onUnmounted } = Vue;

export default {
    template: `
        <div class="ga-page">
            <section class="ga-hero">
                <div>
                    <h2>篮球智能分析中控台</h2>
                    <p>球权分析为主线，按需开启动作高光、指定球员集锦与投篮轨迹预测。</p>
                </div>
                <div class="ga-hero-actions">
                    <el-button type="info" plain @click="openHistory">
                        <i class="el-icon-time"></i> 历史任务
                    </el-button>
                    <el-button type="primary" :loading="submitting" @click="submitTask">
                        开始分析
                    </el-button>
                </div>
            </section>

            <section class="ga-grid">
                <el-card class="ga-panel ga-config">
                    <template #header>
                        <div class="ga-panel-title">任务配置</div>
                    </template>

                    <div class="ga-field">
                        <div class="ga-label">比赛视频</div>
                        <el-upload action="#" :auto-upload="false" :show-file-list="false" :on-change="onVideoChange" accept="video/*">
                            <el-button>选择视频</el-button>
                        </el-upload>
                        <div class="ga-tip" v-if="videoFile">{{ videoFile.name }}</div>
                    </div>

                    <div class="ga-switches">
                        <el-switch v-model="enablePossession" active-text="球权分析" />
                        <el-switch v-model="enableHighlight" active-text="高光剪辑" />
                        <el-switch v-model="enableTrajectory" active-text="轨迹预测" />
                    </div>

                    <div class="ga-field" v-if="enableHighlight">
                        <div class="ga-label">高光模式</div>
                        <el-radio-group v-model="highlightMode">
                            <el-radio-button label="action">动作事件</el-radio-button>
                            <el-radio-button label="player">指定球员</el-radio-button>
                        </el-radio-group>
                    </div>

                    <div class="ga-field" v-if="enableHighlight && highlightMode==='player'">
                        <div class="ga-label">球员查询图</div>
                        <el-upload action="#" :auto-upload="false" :show-file-list="false" :on-change="onQueryChange" accept="image/*">
                            <el-button>上传球员图</el-button>
                        </el-upload>
                        <div class="ga-tip" v-if="queryFile">{{ queryFile.name }}</div>
                    </div>

                    <div class="ga-rim" v-if="enableTrajectory">
                        <div class="ga-label">篮筐标定(可选)</div>
                        <div class="ga-rim-row">
                            <el-input-number v-model="rimCx" :min="0" :controls="false" placeholder="cx" />
                            <el-input-number v-model="rimCy" :min="0" :controls="false" placeholder="cy" />
                            <el-input-number v-model="rimR" :min="1" :controls="false" placeholder="r" />
                        </div>
                    </div>

                    <div class="ga-progress" v-if="taskStatus==='processing'">
                        <el-progress :percentage="progress" :stroke-width="8"></el-progress>
                    </div>

                    <div class="ga-actions">
                        <el-button @click="resetForm">重置</el-button>
                        <el-button type="danger" plain v-if="taskStatus==='processing'" @click="cancelTask">取消任务</el-button>
                    </div>
                </el-card>

                <el-card class="ga-panel ga-stage">
                    <template #header>
                        <div class="ga-panel-title">
                            视频舞台
                            <span class="ga-export">
                                <el-radio-group v-model="stageViewMode" size="mini" @change="onStageViewChange">
                                    <el-radio-button label="auto">综合视频</el-radio-button>
                                    <el-radio-button label="highlight">高光</el-radio-button>
                                    <el-radio-button label="trajectory">轨迹</el-radio-button>
                                    <el-radio-button label="tracking">球权跟踪</el-radio-button>
                                    <el-radio-button label="raw">原视频</el-radio-button>
                                </el-radio-group>
                            </span>
                        </div>
                    </template>

                    <div class="ga-video-wrap">
                        <video v-if="stageVideoUrl" :src="stageVideoUrl" controls autoplay class="ga-video" @error="onStageVideoError"></video>
                        <el-empty v-else description="暂无可播放视频"></el-empty>
                    </div>
                    <div class="ga-tip" v-if="trajectoryHint">{{ trajectoryHint }}</div>

                    <div class="ga-stats">
                        <div class="ga-stat">
                            <span>事件数</span>
                            <strong>{{ metrics.event_count || 0 }}</strong>
                        </div>
                        <div class="ga-stat">
                            <span>高光片段</span>
                            <strong>{{ metrics.highlight_clip_count || 0 }}</strong>
                        </div>
                        <div class="ga-stat">
                            <span>投篮事件</span>
                            <strong>{{ metrics.shot_count || 0 }}</strong>
                        </div>
                    </div>
                </el-card>

                <el-card class="ga-panel ga-data">
                    <template #header>
                        <div class="ga-panel-title">
                            数据栏
                            <span class="ga-export">
                                <el-button size="small" @click="exportResult('json')" :disabled="taskStatus!=='completed'">导出JSON</el-button>
                                <el-button size="small" @click="exportResult('csv')" :disabled="taskStatus!=='completed'">导出CSV</el-button>
                            </span>
                        </div>
                    </template>

                    <el-collapse v-model="activeDataPanels">
                        <el-collapse-item title="球权时间线" name="timeline">
                            <el-table :data="possessionTimeline" size="small" height="200">
                                <el-table-column prop="start_time" label="开始(s)" width="90" />
                                <el-table-column prop="end_time" label="结束(s)" width="90" />
                                <el-table-column prop="player_id" label="持球人" width="90" />
                                <el-table-column prop="team_id" label="队伍" width="70" />
                                <el-table-column prop="duration_s" label="时长(s)" width="90" />
                            </el-table>
                        </el-collapse-item>

                        <el-collapse-item title="事件列表" name="events">
                            <el-table :data="events" size="small" height="200">
                                <el-table-column prop="type" label="类型" width="120" />
                                <el-table-column prop="time" label="时间(s)" width="90" />
                                <el-table-column prop="from_player_id" label="From" width="80" />
                                <el-table-column prop="to_player_id" label="To" width="80" />
                                <el-table-column prop="confidence" label="置信度" width="90" />
                            </el-table>
                        </el-collapse-item>

                        <el-collapse-item title="高光时间轴" name="clips">
                            <el-table :data="highlightClips" size="small" height="220">
                                <el-table-column prop="label" label="标签" width="130" />
                                <el-table-column prop="start_time" label="开始(s)" width="90" />
                                <el-table-column prop="end_time" label="结束(s)" width="90" />
                                <el-table-column prop="duration" label="时长(s)" width="90" />
                            </el-table>
                        </el-collapse-item>
                    </el-collapse>
                </el-card>
            </section>

            <el-drawer v-model="historyVisible" title="分析历史" size="460px">
                <div class="ga-history-wrap">
                    <el-card v-for="item in historyItems" :key="item.id" class="ga-history-card" @click="loadTask(item.id)">
                        <div class="ga-history-top">
                            <strong>#{{ item.id }}</strong>
                            <el-tag size="small" :type="item.status==='completed'?'success':(item.status==='failed'?'danger':'info')">{{ item.status }}</el-tag>
                        </div>
                        <div class="ga-history-name">{{ item.video_filename }}</div>
                        <div class="ga-history-time">{{ item.created_at }}</div>
                    </el-card>
                    <el-empty v-if="historyItems.length===0 && !historyLoading" description="暂无历史"></el-empty>
                    <div v-loading="historyLoading" style="height: 16px;"></div>
                    <el-pagination
                        small
                        layout="prev, pager, next"
                        :current-page="historyPage"
                        :page-size="historyPageSize"
                        :total="historyTotal"
                        @current-change="onHistoryPageChange"
                    />
                </div>
            </el-drawer>
        </div>
    `,
    setup() {
        const videoFile = ref(null);
        const queryFile = ref(null);
        const enablePossession = ref(true);
        const enableHighlight = ref(true);
        const enableTrajectory = ref(false);
        const highlightMode = ref("action");
        const rimCx = ref(null);
        const rimCy = ref(null);
        const rimR = ref(null);

        const submitting = ref(false);
        const taskId = ref(null);
        const taskStatus = ref("idle");
        const progress = ref(0);
        const taskResult = ref(null);
        const stageVideoIndex = ref(0);
        const stageViewMode = ref("auto");
        let pollTimer = null;

        const historyVisible = ref(false);
        const historyLoading = ref(false);
        const historyItems = ref([]);
        const historyPage = ref(1);
        const historyPageSize = ref(10);
        const historyTotal = ref(0);
        const activeDataPanels = ref(["timeline", "events", "clips"]);

        const metrics = computed(() => taskResult.value?.metrics || {});
        const possessionTimeline = computed(() => taskResult.value?.analysis_bundle?.possession?.timeline || []);
        const events = computed(() => taskResult.value?.analysis_bundle?.events || []);
        const highlightClips = computed(() => taskResult.value?.analysis_bundle?.highlights?.clips || []);
        const trajectorySummary = computed(() => taskResult.value?.analysis_bundle?.trajectory || {});
        const trajectoryHint = computed(() => {
            const t = trajectorySummary.value || {};
            if (!t.enabled) return "";
            if (t.error) return `轨迹子模块状态: ${String(t.error)}`;
            const shotCount = Array.isArray(t.shot_events) ? t.shot_events.length : 0;
            if (shotCount > 0) return "";
            const reason = (t.shot_prediction || {}).reason || "no_shot_event";
            return `轨迹未触发投篮预测，原因: ${String(reason)}`;
        });
        const stageVideoVersion = computed(() => {
            const key = taskResult.value?.finished_at || taskResult.value?.task_id || taskId.value || "";
            return key ? String(key) : "1";
        });
        const _appendVersion = (url) => {
            if (!url) return "";
            const sep = String(url).includes("?") ? "&" : "?";
            return `${url}${sep}v=${encodeURIComponent(stageVideoVersion.value)}`;
        };
        const stageVideoCandidates = computed(() => {
            const art = taskResult.value?.artifacts || {};
            const src = {
                analysis: art.analysis_video_url ? String(art.analysis_video_url) : "",
                highlight: art.highlight_video_url ? String(art.highlight_video_url) : "",
                trajectory: art.trajectory_video_url ? String(art.trajectory_video_url) : "",
                tracking: art.tracking_video_url ? String(art.tracking_video_url) : "",
                raw: art.raw_video_url ? String(art.raw_video_url) : ""
            };
            const mode = String(stageViewMode.value || "auto");
            const ordered = mode === "highlight"
                ? [src.highlight, src.analysis, src.tracking, src.trajectory, src.raw]
                : mode === "trajectory"
                    ? [src.trajectory, src.analysis, src.highlight, src.tracking, src.raw]
                    : mode === "tracking"
                        ? [src.tracking, src.analysis, src.highlight, src.trajectory, src.raw]
                        : mode === "raw"
                            ? [src.raw, src.analysis, src.highlight, src.trajectory, src.tracking]
                            : [src.analysis, src.highlight, src.trajectory, src.tracking, src.raw];
            return [...new Set(ordered.filter(Boolean))].map((u) => _appendVersion(u));
        });
        const stageVideoUrl = computed(() => {
            const list = stageVideoCandidates.value || [];
            if (!list.length) return "";
            const idx = Math.max(0, Math.min(stageVideoIndex.value, list.length - 1));
            return list[idx] || "";
        });

        const onVideoChange = (uploadFile) => {
            videoFile.value = uploadFile.raw;
        };
        const onQueryChange = (uploadFile) => {
            queryFile.value = uploadFile.raw;
        };

        const submitTask = async () => {
            if (!videoFile.value) {
                ElementPlus.ElMessage.warning("请先选择视频");
                return;
            }
            if (enableHighlight.value && highlightMode.value === "player" && !queryFile.value) {
                ElementPlus.ElMessage.warning("指定球员模式需要上传球员图");
                return;
            }
            submitting.value = true;
            try {
                const form = new FormData();
                form.append("video", videoFile.value);
                form.append("enable_possession", String(enablePossession.value));
                form.append("enable_highlight", String(enableHighlight.value));
                form.append("highlight_mode", String(highlightMode.value));
                form.append("enable_trajectory", String(enableTrajectory.value));
                if (queryFile.value) form.append("query", queryFile.value);
                if (enableTrajectory.value && rimCx.value != null && rimCy.value != null && rimR.value != null) {
                    form.append("rim_cx", String(rimCx.value));
                    form.append("rim_cy", String(rimCy.value));
                    form.append("rim_r", String(rimR.value));
                }
                const res = await axios.post("/api/game-analysis/analyze", form, {
                    headers: { "Content-Type": "multipart/form-data" }
                });
                taskId.value = res.data.task_id;
                taskStatus.value = "processing";
                progress.value = 0;
                taskResult.value = null;
                stageVideoIndex.value = 0;
                stageViewMode.value = "auto";
                startPolling();
                ElementPlus.ElMessage.success("任务已提交");
            } catch (err) {
                ElementPlus.ElMessage.error(err.response?.data?.detail || "任务提交失败");
            } finally {
                submitting.value = false;
            }
        };

        const startPolling = () => {
            if (pollTimer) clearInterval(pollTimer);
            pollTimer = setInterval(async () => {
                if (!taskId.value) return;
                try {
                    const res = await axios.get(`/api/game-analysis/task/${taskId.value}`);
                    const st = res.data.status;
                    taskStatus.value = st;
                    progress.value = typeof res.data.progress === "number" ? res.data.progress : progress.value;
                    if (st === "completed") {
                        taskResult.value = res.data;
                        stageVideoIndex.value = 0;
                        stageViewMode.value = "auto";
                        progress.value = 100;
                        clearInterval(pollTimer);
                        pollTimer = null;
                    } else if (st === "failed" || st === "cancelled") {
                        clearInterval(pollTimer);
                        pollTimer = null;
                        ElementPlus.ElMessage.error(res.data.error || "任务失败");
                    }
                } catch (err) {
                    console.error("poll game-analysis error", err);
                }
            }, 2500);
        };

        const cancelTask = async () => {
            if (!taskId.value) return;
            try {
                await axios.post(`/api/game-analysis/cancel/${taskId.value}`);
                ElementPlus.ElMessage.success("已发送取消请求");
            } catch (err) {
                ElementPlus.ElMessage.error("取消失败");
            }
        };

        const resetForm = () => {
            videoFile.value = null;
            queryFile.value = null;
            enablePossession.value = true;
            enableHighlight.value = true;
            enableTrajectory.value = false;
            highlightMode.value = "action";
            rimCx.value = null;
            rimCy.value = null;
            rimR.value = null;
            taskId.value = null;
            taskStatus.value = "idle";
            progress.value = 0;
            taskResult.value = null;
            stageVideoIndex.value = 0;
            stageViewMode.value = "auto";
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
        };

        const onStageViewChange = () => {
            stageVideoIndex.value = 0;
        };

        const onStageVideoError = () => {
            const list = stageVideoCandidates.value || [];
            if (stageVideoIndex.value + 1 < list.length) {
                stageVideoIndex.value += 1;
                ElementPlus.ElMessage.warning("当前视频解码失败，已切换备用视频源");
                return;
            }
            ElementPlus.ElMessage.error("视频无法播放，请重新分析生成新视频");
        };

        const loadHistory = async () => {
            historyLoading.value = true;
            try {
                const res = await axios.get("/api/game-analysis/history", {
                    params: {
                        page: historyPage.value,
                        page_size: historyPageSize.value,
                        include_analysis: false
                    }
                });
                historyItems.value = res.data.items || [];
                historyTotal.value = (res.data.pagination || {}).total || 0;
            } catch (err) {
                ElementPlus.ElMessage.error("加载历史失败");
            } finally {
                historyLoading.value = false;
            }
        };

        const openHistory = async () => {
            historyVisible.value = true;
            await loadHistory();
        };

        const onHistoryPageChange = async (page) => {
            historyPage.value = page;
            await loadHistory();
        };

        const loadTask = async (id) => {
            try {
                const res = await axios.get(`/api/game-analysis/task/${id}`);
                taskId.value = id;
                taskStatus.value = res.data.status;
                if (res.data.status === "completed") {
                    taskResult.value = res.data;
                    stageVideoIndex.value = 0;
                    stageViewMode.value = "auto";
                    progress.value = 100;
                } else {
                    taskResult.value = null;
                    startPolling();
                }
                historyVisible.value = false;
            } catch (err) {
                ElementPlus.ElMessage.error("加载任务详情失败");
            }
        };

        const exportResult = async (fmt) => {
            if (!taskId.value) return;
            try {
                const res = await axios.get(`/api/game-analysis/export/${taskId.value}`, {
                    params: { format: fmt }
                });
                if (res.data?.download_url) {
                    window.open(res.data.download_url, "_blank");
                } else {
                    ElementPlus.ElMessage.warning("导出文件未生成");
                }
            } catch (err) {
                ElementPlus.ElMessage.error("导出失败");
            }
        };

        onUnmounted(() => {
            if (pollTimer) clearInterval(pollTimer);
        });

        return {
            videoFile,
            queryFile,
            enablePossession,
            enableHighlight,
            enableTrajectory,
            highlightMode,
            rimCx,
            rimCy,
            rimR,
            submitting,
            taskStatus,
            progress,
            taskResult,
            metrics,
            possessionTimeline,
            events,
            highlightClips,
            trajectoryHint,
            stageViewMode,
            stageVideoUrl,
            onStageViewChange,
            onStageVideoError,
            onVideoChange,
            onQueryChange,
            submitTask,
            cancelTask,
            resetForm,
            exportResult,
            historyVisible,
            historyLoading,
            historyItems,
            historyPage,
            historyPageSize,
            historyTotal,
            activeDataPanels,
            openHistory,
            onHistoryPageChange,
            loadTask
        };
    }
};
