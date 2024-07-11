/**
 * 用于配置跨域隔离，便于多线程运算
 */

const express = require('express');
const path = require('path');
const app = express();
const port = 8080;
 
// 设置 Cross-Origin-Opener-Policy 和 Cross-Origin-Embedder-Policy 响应头
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});

// 提供静态文件服务
app.use(express.static(path.join(__dirname, 'public')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});