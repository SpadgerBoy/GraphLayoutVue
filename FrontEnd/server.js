// const express = require('express');
// const path = require('path');
// const app = express();
// const port = 8080;

// // 设置 Cross-Origin-Opener-Policy 和 Cross-Origin-Embedder-Policy 响应头
// app.use((req, res, next) => {
//   res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
//   res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
//   next();
// });

// // 提供静态文件服务（假设静态文件在 'public' 目录中）
// app.use(express.static(path.join(__dirname, 'public')));

// app.get('/', (req, res) => {
//   res.sendFile(path.join(__dirname, 'public', 'index.html'));
// });

// app.listen(port, () => {
//   console.log(`Server is running at http://localhost:${port}`);
// });


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