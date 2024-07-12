/* 
 * 将num_nodes, all_adges传入服务器，并从服务器获取返回的数据 
 */

import axios from 'axios';

export default async function getData(num_nodes, all_adges) {

    const datajson = {
        data1: num_nodes,
        data2: all_adges
      };

    try {
      const response = await axios.post('http://101.36.73.174:14449/getdata', datajson);
      console.log('Data sent successfully:', response.data);
      var responseMessage = response.data; 
      // console.log(responseMessage);
    } catch (error) {
      console.error('Error sending data:', error);
      var responseMessage = 'Error sending data'; 
    }

    return responseMessage;
}
