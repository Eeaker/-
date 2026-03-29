# BallShow Web 前端开发文档 (Frontend)

## 1. 架构总览
Web 端基于 **Vue.js 3** 和 **Element Plus** 构建。因这套系统主要是用做 AI 大屏展示和快速原型验证，所以没有采用复杂的工程化构建（Webpack / Vite 等），而是直接通过 CDN/本地 Lib 引入 `vue.global.prod.min.js`, 实现了“单文件应用”式开发。代码轻巧、渲染迅速。

## 2. 核心技术栈
- **UI 框架**: Vue 3 (Composition API 写法)
- **UI 组件库**: Element Plus (提供基础的数据驱动视图组件、表格、抽屉和弹窗等)
- **数据可视化**: ECharts (Apache ECharts, 用于渲染“特征聚类分布”之类的统计报表)
- **请求库**: Axios
- **CSS 风格**: 深色主题定制 (`#0D0D12` 极简暗黑风，配合橙色高亮强调运动竞技感)

## 3. 功能模块
入口页面 `index.html` 负责加载各依赖和组件。功能具体由不同子页签 (Tabs) 承载：

### 3.1 首页监控大盘 (Dashboard)
- 对应组件 `Dashboard.js`
- 挂载 ECharts 图表，动态加载接口 `/api/dashboard` 以获取如下数据：
  1. 系统服务状态与 GPU (Online/Offline) 显示。
  2. TransReID 双模型各项结构参数。
  3. 累计搜索次数与检索成功率。

### 3.2 球员检索 (Image ReID)
- 对应组件 `ImageSearch.js`
- **流程**:
  1. 用户上传某路人拍下的球员图片。
  2. Vue 前端构造 `FormData` POST 到 `/api/reid/search`，可配置 Top-K 大小。
  3. 收取特征比对列表并在画廊瀑布流(Gird)中可视化渲染评分、Rank、及其原图长相。

### 3.3 赛事分析 (Video Action)
- 对应组件 `VideoAnalysis.js`
- **流程**: 
  1. 上传“追踪目标人脸”图片和“比赛长录像(MP4)”。
  2. 调用后端异步视频处理接口。
  3. 前端长轮询机制探测 `progress`。
  4. 渲染任务报告、出场时刻表和聚合好的**高光视频合集 (Highlight Video)** 播放器。

## 4. 运行方式
**前端完全静态化**，没有 Node.js 开发流。
请任意开启一个本地静态文件 HTTP 服务器即可访问，或者如果您已启动了 FastAPI 后端，会自动通过内置静态挂载分发这个目录：
URL: `http://127.0.0.1:8000/`

## 5. 目录结构
```text
platform/frontend/
├── index.html                  # 站点骨架入口
├── css/
│   └── dark-theme.css          # 全局深色主题重写样式
├── js/
│   ├── app.js                  # Vue 主实例和路由控制
│   ├── api.js                  # Axios 配置，包含 Token 自动携带逻辑
│   └── components/
│       ├── AppLayout.js        # 侧边栏/顶栏整体布局
│       ├── Dashboard.js        # 实时大屏页面
│       ├── ImageSearch.js      # 图片球员检索
│       ├── VideoAnalysis.js    # 视频高光提取
│       └── History.js          # 查询记录看板
└── lib/                        # 无网离线运行使用的压缩版 JS/CSS 包（包括 Vue, Echarts, Axios 等）
```
