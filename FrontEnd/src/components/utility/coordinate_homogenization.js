/** 
 * 将坐标间距均匀化  
*/
export default function convertGridToCart(pos, level) {
    /*
    Args:
        pos: [[x,y],[x,y],[x,y],...]
        level: [1,1,2,2,3,]

    Returns:
        pos: [[x,y],[x,y],[x,y],...]
    */

    let newPos = new Array(pos.length); // 初始化一个与pos长度相同的数组

    let levelNode = [];
    let num = 0;
    let iLevelNodesPos = [];
    let flag = 1;

    // console.log(level.length, pos.length);
    for (let index = 0; index < level.length; index++) {
        let i = level[index];
        if (i === flag) {
            iLevelNodesPos.push(pos[index][0]);
        }

        if (i !== flag || index === level.length - 1) {
            let sortedIndices = iLevelNodesPos.map((val, idx) => [val, idx])
                                               .sort((a, b) => a[0] - b[0])
                                               .map(item => item[1]);
            let rankedIndices = new Array(sortedIndices.length);
            sortedIndices.forEach((j, index0) => {
                rankedIndices[index0] = j + num;
            });
            levelNode.push(rankedIndices);
            
            num += iLevelNodesPos.length;
            iLevelNodesPos = [pos[index][0]];
            flag += 1;
        }
    }
    // console.log('levelNode',levelNode);

    // get each level max width and overall max width
    let maxW = Math.max(...levelNode.map(l => l.length));
    let xstep = 1.0 / maxW;  // abstand zwischen den spalten
    let ystep = 1.0 / levelNode.length;

    let y = ystep / 2.0;
    for (let l of levelNode) {  // for each level l in level
        let x = xstep / 2.0 + (maxW - l.length) / (maxW * 2.0);  // start pos and centering + padding and centering

        for (let i = 0; i < l.length; i++) {
            let n = l[i];       
            newPos[n] = [x, y]; 
            x += xstep;         
        }
        y += ystep;
    }

    return newPos;
};


