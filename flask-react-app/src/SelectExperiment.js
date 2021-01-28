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


class SelectExperiment extends React.Component {


  constructor() {
    super();
    this.state = {
      originHashtags: '', 
      campaignName: '',
      experiments: []
    };
    this.getExperiment = this.getExperiment.bind(this)

  }



  componentWillMount() {

    fetch('/experiments').then(res => res.json()).then(data => {
      console.log(data)
      this.setState({'experiments': data.result});
    });
    
  }

  componentDidMount() {

}







  getExperiment(experiment_id) {
    let server_url = 'http://127.0.0.1:8000/get_experiment'

    const server_headers = {
      'Accept': '*/*',
      'Content-Type': 'application/json',
      "Access-Control-Origin": "*",
      "Access-Control-Request-Headers": "*",
      "Access-Control-Request-Method": "*",
      "Connection":"keep-alive"
    }


    fetch(server_url,
      {
          headers: server_headers,
          method: "GET"
      })
      .then(res=>{ return res.json()})
      .then(data => {
        this.props.history.push('/experiments/'+experiment_id)
        document.location.reload()
      })
      .catch(res=> console.log(res))
  
  
   } 
  


   render() {
  return (
    <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}}>
       

        <List
    itemLayout="horizontal"
    dataSource={this.state.experiments}
    bordered
    style={{'width': '500px'}}
    renderItem={experiment => (
      <List.Item>
        <List.Item.Meta
          avatar={<Avatar src="https://img.icons8.com/wired/64/000000/thin-test-tube.png" />}
          title={<a href="https://ant.design">{experiment.name}</a>}
          description={experiment.experiment_id}
          onClick={() => this.getExperiment(experiment.experiment_id)}
        />
      </List.Item>)}
      />


    </div>
  )
   }
}




export default SelectExperiment;
