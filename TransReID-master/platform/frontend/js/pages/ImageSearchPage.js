const { ref, computed } = Vue;

export default {
    template: `
        <div style="padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2 style="margin: 0; color: var(--text-main); font-weight: 700; letter-spacing: -0.02em;">AI 图片识人 (Image ReID)</h2>
            </div>
            
            <el-row :gutter="20">
                <!-- 左侧：上传区 -->
                <el-col :span="8">
                    <el-card style="background: var(--bg-overlay); border-color: var(--border-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-soft);">
                        <template #header>
                            <div style="font-weight: bold;">上传目标图 (Query)</div>
                        </template>
                        <el-upload
                            class="upload-demo"
                            drag
                            action="#"
                            :auto-upload="false"
                            :show-file-list="false"
                            :on-change="onFileChange"
                        >
                            <el-icon class="el-icon--upload"><i class="el-icon-upload"></i></el-icon>
                            <div class="el-upload__text">将图片拖到此处，或 <em>点击选取</em></div>
                        </el-upload>

                        <!-- 预览图 -->
                        <div v-if="previewUrl" style="margin-top: 20px; text-align: center;">
                            <img :src="previewUrl" style="max-width: 100%; max-height: 300px; border-radius: 8px; border: 2px solid var(--primary-color);">
                            <div style="margin-top: 20px;">
                                <el-form-item label="检索数量 (Top-K)">
                                    <el-slider v-model="topk" :min="1" :max="100" show-input></el-slider>
                                </el-form-item>
                                <el-button type="primary" size="large" style="width: 100%; font-weight: bold;" @click="submitSearch" :loading="loading">
                                    开始全库检索 🚀
                                </el-button>
                            </div>
                        </div>
                    </el-card>
                </el-col>

                <!-- 右侧：结果展示区 -->
                <el-col :span="16">
                    <el-card class="result-card-scrollable" style="background: var(--bg-overlay); border-color: var(--border-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-soft); height: calc(100vh - 120px); overflow-y: auto;">
                        <template #header>
                            <div style="font-weight: bold;">
                                检索结果 (Gallery)
                                <el-tag v-if="results.length" type="success" style="margin-left: 10px;">耗时: < 0.5s</el-tag>
                            </div>
                        </template>
                        
                        <div v-if="!results.length && !loading" style="text-align: center; color: #666; padding: 100px 0;">
                            上传图片后点击检索
                        </div>
                        
                        <div v-if="loading" style="text-align: center; color: var(--primary-color); padding: 100px 0;">
                            <el-spinner></el-spinner>
                            <div style="margin-top: 15px; font-weight: 500;">双 ViT 模型矩阵高速运算中...</div>
                        </div>

                        <div class="result-grid" v-if="results.length">
                            <div v-for="(res, index) in results" :key="res.rank" class="result-item">
                                <div class="result-img-wrapper" style="overflow: hidden;">
                                    <el-image 
                                        :src="res.path" 
                                        class="result-img"
                                        fit="cover"
                                        style="cursor: pointer;"
                                        @click="openViewer(index)"
                                    ></el-image>
                                    <div class="score-badge">相似度 {{(res.score * 100).toFixed(1)}}%</div>
                                </div>
                                <div class="result-info">
                                    <div style="font-weight: bold; color: var(--text-main);">Rank {{res.rank}}</div>
                                    <div style="color: #888; font-size: 12px; margin-top: 5px;">ID: {{res.person_id}}</div>
                                    <div style="color: #666; font-size: 10px; margin-top: 2px; word-break: break-all;">
                                        {{res.filename}}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
            
            <!-- 全局独立的大图预览器，彻底脱离原父级 DOM 结构限制 -->
            <el-image-viewer 
                v-if="showViewer" 
                :url-list="previewSrcList" 
                :initial-index="viewerIndex" 
                @close="showViewer = false" 
                :z-index="9999">
            </el-image-viewer>
        </div>
    `,
    setup() {
        const file = ref(null);
        const previewUrl = ref('');
        const topk = ref(10);
        const loading = ref(false);
        const results = ref([]);

        const onFileChange = (uploadFile) => {
            file.value = uploadFile.raw;
            previewUrl.value = URL.createObjectURL(uploadFile.raw);
            results.value = []; // 清空上次结果
        };

        const submitSearch = async () => {
            if (!file.value) return;
            loading.value = true;
            results.value = [];

            const formData = new FormData();
            formData.append('file', file.value);
            formData.append('topk', topk.value);

            try {
                const res = await axios.post('/api/reid/search', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                results.value = res.data.results;
                if (results.value.length > 0) {
                    ElementPlus.ElMessage.success(`检索完成，返回 Top-${results.value.length}，最高相似度 ${(results.value[0].score * 100).toFixed(1)}%`);
                }
            } catch (err) {
                ElementPlus.ElMessage.error('检索失败，请检查模型是否启动');
            } finally {
                loading.value = false;
            }
        };

        const previewSrcList = computed(() => {
            return results.value.map(res => res.path);
        });

        const showViewer = ref(false);
        const viewerIndex = ref(0);

        const openViewer = (index) => {
            viewerIndex.value = index;
            showViewer.value = true;
        };

        return { file, previewUrl, topk, loading, results, onFileChange, submitSearch, previewSrcList, showViewer, viewerIndex, openViewer };
    }
};
