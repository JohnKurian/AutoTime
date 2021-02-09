import logo from './logo.svg';
import React, { useState, useEffect } from 'react';
import './App.css';

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
  Redirect
} from "react-router-dom";
import { render } from '@testing-library/react';

import {
  ExperimentOutlined,
  HomeOutlined ,
  SettingFilled,
  TeamOutlined,
  UserOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

import { Upload, message, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';


import { List, Avatar } from 'antd';

const data = [
  {
    title: 'Ant Design Title 1',
  },
  {
    title: 'Ant Design Title 2',
  },
  {
    title: 'Ant Design Title 3',
  },
  {
    title: 'Ant Design Title 4',
  },
];

const uploadprops = {
  name: 'file',
  action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
  headers: {
    authorization: 'authorization-text',
  },
  onChange(info) {
    if (info.file.status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
};


class Datasets extends React.Component {


  constructor() {
    super();
    this.state = {
      originHashtags: '', 
      campaignName: '',
      experiments: []
    };
    this.getDatasetInfo = this.getDatasetInfo.bind(this)

  }


  getDatasetInfo() {

  }



  componentWillMount() {

    fetch('/datasets').then(res => res.json()).then(data => {
      console.log(data)
      this.setState({'datasets': data.datasets});
    });
    
  }

  componentDidMount() {

}


  


   render() {
  return (
    <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>

<div style={{'marginTop': '50px', 'marginBottom': '50px'}}>
<Upload {...uploadprops}>
    <Button icon={<UploadOutlined />}>Upload dataset</Button>
  </Upload>
  </div>
       

        <List
    itemLayout="horizontal"
    dataSource={this.state.datasets}
    bordered
    style={{'width': '500px'}}
    renderItem={dataset => (
      <List.Item>
        <List.Item.Meta
          avatar={<DatabaseOutlined />}
          title={<a href="https://ant.design">{dataset}</a>}
          description={''}
          onClick={() => this.getDatasetInfo()}
        />
      </List.Item>)}
      />


    </div>
  )
   }
}




export default Datasets;
