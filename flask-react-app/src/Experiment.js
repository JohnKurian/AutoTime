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

import { Typography } from 'antd';

import { Button, Tooltip } from 'antd';
import { BarChartOutlined, RocketOutlined, PlayCircleOutlined } from '@ant-design/icons';

import { List, Avatar } from 'antd';

const { Title } = Typography;


class Experiment extends React.Component {


  constructor() {
    super();
    this.state = {
      originHashtags: '', 
      campaignName: '',
      experiments: []
    };
    this.getRun = this.getRun.bind(this)
    this.startRun = this.startRun.bind(this)

  }



  componentWillMount() {


    
  }

  componentDidMount() {
      console.log('props', this.props)

      let experiment_id = this.props.history.location.pathname.split('/')[2]

    fetch('/get_runs?experiment_id='+experiment_id).then(res => res.json()).then(data => {
        console.log(data)
        this.setState({'runs': data.result}); 
      });

}


startRun(event) {
    event.preventDefault();
    console.log(event)
    let experiment_id = this.props.history.location.pathname.split('/')[2]
    let server_url = 'http://127.0.0.1:8000/create_run?experiment_id=' + experiment_id

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
          method: "POST",
          body: JSON.stringify({'exp_name': experiment_id})
      })
      .then(res=>{ return res.json()})
      .then(data => {
        //this.props.history.push('/experiments/'+experiment_id)
        //document.location.reload()
      })
      .catch(res=> console.log(res))
  
  
   } 







getRun(run_id) {
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
        this.props.history.push('/runs/'+run_id)
        document.location.reload()
      })
      .catch(res=> console.log(res))
  
  
   } 
  


   render() {
  return (
    <div style={{'display': 'flex', 'flexDirection': 'column'}}>
        <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}}>
          <Title level={3}>This_is_new</Title>
          <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>ID:</Title> 2</div>
          <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Dataset location:</Title> data.csv</div>
          <Title style={{'marginTop': '20px'}} level={4}>Best run details</Title>
          <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Run ID:</Title> dsfs342349ßs0dfsdfssdf</div>
          <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>r2</Title> 0.94</div>
          <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Model:</Title> LSTM</div>
        </div>

        <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline', 'marginTop': '50px'}}>
          <Button style={{'marginRight': '25px'}} type="primary" onClick={this.startRun} icon={<PlayCircleOutlined />}>Start new run</Button>
          <Button style={{'marginRight': '25px'}} type="primary" onClick={this.startRun} icon={<RocketOutlined />}>Deploy model</Button>
          <Button style={{'marginRight': '25px'}} type="primary" onClick={this.startRun} icon={<BarChartOutlined />}>Generate report</Button>
         </div>


         <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline', 'marginTop': '50px'}}>
         <Title style={{'marginTop': '20px'}} level={4}>Runs</Title>
         </div>

      
      <List
        itemLayout="horizontal"
        dataSource={this.state.runs}
        bordered
        style={{'width': '500px'}}
        renderItem={run => (
      <List.Item>
        <List.Item.Meta
          avatar={<img src="https://img.icons8.com/metro/26/000000/start.png"/>}
          title={<a href="https://ant.design">{run.name}</a>}
          description={run.run_id}
          onClick={() => this.getRun(run.run_id)}
        />
      </List.Item>)}
      />
    </div>
  )
   }
}




export default Experiment;
