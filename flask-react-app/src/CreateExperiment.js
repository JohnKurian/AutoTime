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

import { Form, Input, InputNumber, Button } from 'antd';


const layout = {
  labelCol: { span: 8 },
  wrapperCol: { span: 16 },
};

const validateMessages = {
  required: '${label} is required!',
  types: {
    email: '${label} is not a valid email!',
    number: '${label} is not a valid number!',
  },
  number: {
    range: '${label} must be between ${min} and ${max}',
  },
};


class CreateExperiment extends React.Component {


  constructor() {
    super();
    this.state = {
      originHashtags: '', 
      campaignName: '',
      experiments: [],
      exp_name: ''
    };

    this.handleChange = this.handleChange.bind(this)
    this.createExperiment = this.createExperiment.bind(this)

  }



  componentWillMount() {


    
  }

  componentDidMount() {

}


handleChange(evt) {
  this.setState({exp_name: evt.target.value})
}


onFinish(values) {
  console.log(values);

  let server_url = 'http://127.0.0.1:8000/create_experiment'

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
        body: JSON.stringify({
          'exp_name': values['experiment']['name'], 
          'dataset_location': values['experiment']['dataset_location'],
          'notes': values['experiment']['notes']
      })
    })
    .then(res=>{ return res.json()})
    .then(data => {
      //this.props.history.push('/experiments/'+experiment_id)
      //document.location.reload()
    })
    .catch(res=> console.log(res))


};







  createExperiment(event) {
    event.preventDefault();
    console.log(event)
    let server_url = 'http://127.0.0.1:8000/create_experiment'

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
          body: JSON.stringify({'exp_name': this.state.exp_name})
      })
      .then(res=>{ return res.json()})
      .then(data => {
        //this.props.history.push('/experiments/'+experiment_id)
        //document.location.reload()
      })
      .catch(res=> console.log(res))
  
  
   } 
  


   render() {
    return (
      <div style={{'display': 'flex', 'flexDirection': 'column', 'width': '500px'}}>


          <Form {...layout} name="nest-messages" onFinish={this.onFinish} validateMessages={validateMessages}>
      <Form.Item name={['experiment', 'name']} label="Experiment Name" rules={[{ required: false }]}>
        <Input />
      </Form.Item>
      <Form.Item name={['experiment', 'dataset_location']} label="Dataset location" rules={[{ required: false }]}>
        <Input />
      </Form.Item>
      <Form.Item name={['experiment', 'notes']} label="Notes">
        <Input.TextArea />
      </Form.Item>
      <Form.Item wrapperCol={{ ...layout.wrapperCol, offset: 8 }}>
        <Button type="primary" htmlType="submit">
          Submit
        </Button>
      </Form.Item>
    </Form>
  
      </div>
    );
   }
}



export default CreateExperiment;
