# BallShow 微信小程序端开发文档 (WeChat MiniProgram)

## 1. 架构总览
专为微信小程序生态量身定制原生版本（纯 WXML + WXSS + JS 构建，无引入 UniApp 或 Taro）。结合 iOS 和 Android 的底层同层渲染差异，对特定组件做出了深度优化。

## 2. 重点解决的技术难题与特性

### 2.1 高光视频时间轴交互
- 在页面 `video_player` 中，直接在 `<video>` 组件下方实现了**横向滚动片段集锦卡片**。
- `bindtimeupdate` 和 `bindended` 实时挂钩，同步播放状态数据到页面 ViewModel。
- **自定义 Seek (`videoContext.seek`)**: 关联高光视频合并时的**时间偏移累加量(`highlight_time`)**，使用户点击底层不同片段就能迅速在混合视频内跳转到位。

### 2.2 两阶段异步轮询长任务
- 微信小程序的 `wx.request` 限制最长不可超 60 秒（通常 30s 即被杀进程或降权）。
- 因此，对于 AI 视频模型处理时常 1-5 分钟的长推流机制，必须在小程序端写：
  - `<Upload Stage 1>` 图片识别素材上传。
  - `<Upload Stage 2>` 视频文件流式上传。
  - `<Task UUID Pending>` 唤醒 `setInterval(3000ms)` 开始轮询进度状态。
  - 最终状态扭转为 "completed" 读取 `analysis_segments` 对象。

### 2.3 网络兼容安全（免配合法域名的开发技巧）
- 将本地开发或局域网局域内通信 (IPv4 Port 8000) 利用局域网特性访问（如 `http://192.168.x.x:8000/api/...`）。
- 为了避开预览调试包不能读取本地临时多媒体文件的限制，对 `video.src` 以及所有 API Domain 配置了网络权限兼容策略。

## 3. 页面模块

### 3.1 `pages/index/` (首页)
展示统计大盘、模型运行状态图和最近记录，UI 复刻 Web 的流光溢彩暗黑运动风格。

### 3.2 `pages/image_search/` (人像找图)
提供小程序接口 `wx.chooseMedia` 拍摄或加载相册。将结果展现在 `scroll-view` 网格中。

### 3.3 `pages/video_analysis/` (比赛高光挖掘)
双料上传图片与视频，动态显示 1% -> 100% 处理流水线圆环进度。分析成功自动弹出。

### 3.4 `pages/video_player/` (原生视频播放器深度页)
基于微信原生 `live-player` 和同层渲染技术，解决原生元素覆盖非原生元素的 `z-index` 失控问题。

## 4. 目录结构
```text
wechat_miniprogram/
├── app.js / app.json / app.wxss      # 全局配置入口与样式 (暗黑风格基调)
├── api.js                            # Request 封装请求类，统一抛出拦截鉴权错误
├── pages/                            # 子页面模块
│   ├── login/                        # 授权注册页
│   ├── index/                        # 综合 Dashboard大盘
│   ├── image_search/                 # 图片智能以图搜人
│   ├── video_analysis/               # 多模态上传等待页
│   ├── video_player/                 # AI剪辑成品播放器
│   └── history/                      # 我的追踪档案历史
```

## 5. 调试方法
1. 下载微信小程序官方「微信开发者工具」。
2. 【项目导入】选中 `platform/wechat_miniprogram/` 目录。
3. 如果未开通企业主体，可填入你自己的测试号 AppID；并在 `详情(Details) -> 本地设置(Local Settings)` 中勾选：
   ✅ **"不校验合法域名、web-view（业务域名）、TLS版本以及HTTPS证书"**
4. 编辑 `api.js` 将基地址 `BASE_URL` 改为你本机后台暴露的局域网 IP (例如 `192.168.31.x`) 即可手机同局域网无缝扫码体验真实效果。
