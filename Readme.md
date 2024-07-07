### 一、BackEnd:

#### 1.相关依赖：

相关依赖的版本：

```bash
conda create -n DiffModel python=3.8

conda activate DiffModel

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

根据requirements.txt安装相关依赖，

```bash
pip install -r requirements.txt
```

安装完对应版本pytorch后可安装requirements/文件夹下的四个轮子.whl

```bash
pip install *.whl
```



#### 2.配置IP和端口：

在main.py中配置服务器对应的IP和端口：

```python
if __name__ == "__main__":
    print('run xx.xx.xx.xx:xxxx')
    app.run(host='xx.xx.xx.xx', port=xxxx)

```



#### 3.run code:

处理来来自前端的数据

```bash
python main.py
```

此部分涉及到稀疏矩阵等的运算，只能放在后端





### 二、FrontEnd:

安装**pnpm**:

```bash
npm install pnpm -g
```

安装依赖包:

```
pnpm i
```

运行:

```bash

pnpm run start

```

tips：

#### 1.跨域隔离

前端中涉及到模型多线程运算，所以需要开启跨域隔离

vue.config.js:

```js
module.exports = {
  outputDir: 'public',

  // 跨域隔离
  devServer: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    }
  }
};
```

server.js:

```js

const express = require('express');
const path = require('path');
const app = express();
const port = 8080;		//与前端端口对应，一般为8080

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
```

#### 2.配置接收服务器的IP与端口：

运行前首先将.\src\components\data_process\get_data.js中IP与端口修改为服务器端的IP与端口

```js
const response = await axios.post('http://xx.xx.xx.xx:xxxx/getdata', datajson);
```



#### 3.文件./src/APP.vue

从一个txt文件中读取节点数量N和图的所有边all_edges，该文件将调用./src/components/run.js处理这些参数



#### 4.文件./src/components/run.js

配置模型的参数：

```js
const config = {
    beta_start: 0.001,
    beta_end: 0.02,
    steps: 200,
    model: "./onnx/GLM200.onnx",
};


const config = {
  beta_start: 0.001,
  beta_end: 0.02,
  steps: 100,
  model: "./onnx/GLM100.onnx",
};
```

其中steps和model为对应的扩散步数和相应的onnx路径，可以选择step=200和step=100两种，**step=200时效果更好但时间更长**



get_new_graph()函数调用.\src\components\data_process\get_data.js,首先将N和all_edges传给后端，接收所需的下列参数：

```js
  const node_emb = repoense.node_emb;
  const node_level = repoense.node_level;
  var pos_init = repoense.pos_init;
  const edge_index = repoense.edge_index;
  const edge_type = repoense.edge_type;
```

然后将这些参数传入DiffModel()，运算得到新的pos，并经过convertGridToCart()对pos进行间距重整。



#### 5.文件./src/components/Vis.vue

根据all_edges和新的pos画图

