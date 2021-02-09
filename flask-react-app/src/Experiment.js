import logo from './logo.svg';
import React, { useState, useEffect } from 'react';
import './App.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dark } from 'react-syntax-highlighter/dist/esm/styles/prism';

import Plot from 'react-plotly.js'
import { Table, Tag, Space } from 'antd';


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

    let experiment_id = this.props.history.location.pathname.split('/')[2];

    fetch('/get_experiment?experiment_id='+experiment_id).then(res => res.json()).then(data => {
      console.log(data)
      this.setState({'exp_details': data.payload}); 
    })


    
  }

  componentDidMount() {
      console.log('props', this.props)

      let experiment_id = this.props.history.location.pathname.split('/')[2];


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
    let server_url = 'http://127.0.0.1:8000/experiments'

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
    <div>
    {this.state.exp_details && <div style={{'display': 'flex', 'flexDirection': 'column'}}>

        <div  style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start'}}>
          <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'flex': 1}}>
            <Title level={3}><Avatar src="https://img.icons8.com/wired/64/000000/thin-test-tube.png" /> {this.state.exp_details['name']}</Title>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>ID:</Title>{this.state.exp_details['experiment_id']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Dataset location:</Title>{this.state.exp_details['dataset_location']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Forecasting horizon:</Title>{this.state.exp_details['forecasting_horizon']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Mode:</Title>{this.state.exp_details['mode']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Predictor Column:</Title>{this.state.exp_details['predictor_column']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Selected algorithms:</Title>{this.state.exp_details['selected_algos']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Notes:</Title>{this.state.exp_details['notes']}</div>

          </div>
          <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'flex': 1}}>
            <Title style={{'marginTop': '20px'}} level={4}>Best run details</Title>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Run ID:</Title>{this.state.exp_details['best_run_id']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>r2</Title> {this.state.exp_details['r2']}</div>
            <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline'}}><Title level={5}>Model:</Title> {this.state.exp_details['model']}</div>
            
            <div style={{'display': 'flex', 'flexDirection': 'column', 'alignItems': 'baseline', 'marginTop': '50px', 'width': '600px'}}>

            <div style={{'width': '600px'}}>
            <SyntaxHighlighter language="javascript" wrapLines={true} style={dark}>
      {'curl -X POST "http://127.0.0.1:8080/7f8608e2-14c6-4071-864c-9ae5306324fd/best_model/predict" \
-H  "accept: application/json" \
-H  "Content-Type: application/json" \
-d "{\"data\": [4.6,11.1,8.7,10,11.3,10.5,9.9,11,14,9.2,9.8,6,9.8,9.2,11.8,10.3,7.5,7.7,15.8,14.6,10.5,11.3]}"'}
    </SyntaxHighlighter>
    </div>
          <a target="_blank" rel="noopener noreferrer" href="http://localhost:8080/docs">
          <Button style={{'marginRight': '25px'}} type="primary" onClick={this.startRun} icon={<RocketOutlined />}>Test model</Button>
          </a>
         </div>

          </div>
        </div>

        <div style={{'display': 'flex', 'flexDirection': 'row', 'alignItems': 'baseline', 'marginTop': '50px'}}>
        <a target="_blank" rel="noopener noreferrer" href="http://localhost:3000">
          <Button style={{'marginRight': '25px'}} type="primary" onClick={() => {}} icon={<BarChartOutlined />}>Open MlFlow backend</Button>
          </a>
         </div>



        
      

      <div style={{'display': 'flex'}} >
        <div style={{'display': 'flex', 'flex': 1, 'flexDirection': 'column'}}>

        <div style={{'marginTop': '50px'}}>
         <Title style={{'marginTop': '20px'}} level={4}>Runs</Title>
         </div>

         <Plot
        data={[
          {
            y: this.state.exp_details['all_r2'],
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: 'blue'},
          }
        ]}
        layout={ {width: 520, height: 360, title: 'R2 curve'} }
      />


<div style={{'marginTop': '50px'}}>
<Button style={{'marginRight': '25px'}} type="primary" onClick={this.startRun} icon={<PlayCircleOutlined />}>Start new run</Button>
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

      <div style={{'display': 'flex', 'flex': 2}}>

      <Table scroll={{ x: 'max-content' }} style ={{'width': '700px', 'marginTop': '100px'}} columns={this.state.exp_details['data_columns']} dataSource={this.state.exp_details['datasource']} />

      </div>

    </div>

    </div>
    
    }


   
    </div>
  )
   }
}




export default Experiment;
