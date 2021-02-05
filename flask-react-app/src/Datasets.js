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
