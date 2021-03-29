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

import { Select } from 'antd';

import { message } from 'antd';


const { Option } = Select;


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
      exp_name: '',
      datasets: [],
      dataset_columns: [],
      forecasting_horizon: 10,
      selected_algorithms: [],
      available_algorithms: ['TCN', 'LSTM', 'Sktime-RandomForest', 'Sktime-KNN', 'Sktime-ThetaForecaster', 'NBeats'],
      available_modes: ['univariate', 'multivariate'],
      selected_mode: ['univariate']
    };

    this.handleChange = this.handleChange.bind(this)
    this.createExperiment = this.createExperiment.bind(this)
    this.onFinish = this.onFinish.bind(this)
    this.onModeChange = this.onModeChange.bind(this)
    this.onDatasetChange = this.onDatasetChange.bind(this)
    this.onPredictorColumnChange = this.onPredictorColumnChange.bind(this)
    this.handleMultiSelectChange = this.onPredictorColumnChange.bind(this)
  

  }

  onInputNumberChange(value) {
    console.log('changed', value);
  }

  onModeChange(value) {
    console.log('changed', value);
  }

  onPredictorColumnChange(value) {
    console.log('changed', value);
  }


  onDatasetChange(value) {
    console.log(`selected ${value}`);
    fetch('/datasets/info?data='+value).then(res => res.json()).then(data => {
      console.log(data)
      this.setState({
        'dataset_columns': data.dataset_columns
    });
    });
  }
  
  onBlur() {
    console.log('blur');
  }
  
  onFocus() {
    console.log('focus');
  }
  
  onSearch(val) {
    console.log('search:', val);
  }

  success = () => {
    message.success('This is a success message');
  }



  componentWillMount() {

    fetch('/datasets').then(res => res.json()).then(data => {
      console.log(data)
      this.setState({'datasets': data.datasets});
    });
    
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
          'forecasting_horizon': values['experiment']['forecasting_horizon'],
          'mode': values['experiment']['mode'],
          'predictor_column': values['experiment']['predictor_column'],
          'selected_algos': values['experiment']['selected_algos'],
          'notes': values['experiment']['notes']
      })
    })
    .then(res=>{ return res.json()})
    .then(data => {
      message.success('Successfully created experiment');
      this.props.history.push('/experiments/'+data['experiment_id'])
      document.location.reload()
    })
    .catch(res=> console.log(res))


};


onFinishTemp(values) {
  console.log(values);
}

handleMultiSelectChange(value) {
  console.log(`selected ${value}`);
}







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
        message.success('Successfully created experiment');
        this.props.history.push('/experiments/'+data['experiment_id'])
        document.location.reload()
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
      
      <Select
        showSearch
        style={{ }}
        placeholder="Select a dataset"
        optionFilterProp="children"
        onChange={this.onDatasetChange}
        onFocus={this.onFocus}
        onBlur={this.onBlur}
        onSearch={this.onSearch}
        filterOption={(input, option) =>
          option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
        }
      >


      {this.state.datasets.map((dataset) =>
      <Option value={dataset}>{dataset}</Option>
      )}

      </Select>

      </Form.Item>

      <Form.Item name={['experiment', 'predictor_column']} label="Predictor column" rules={[{ required: false }]}>
      
      <Select
        showSearch
        style={{ }}
        placeholder="Select a predictor column"
        optionFilterProp="children"
        onChange={this.onPredictorColumnChange}
        onFocus={this.onFocus}
        onBlur={this.onBlur}
        onSearch={this.onSearch}
        filterOption={(input, option) =>
          option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
        }
      >


      {this.state.dataset_columns.map((column) =>
      <Option value={column}>{column}</Option>
      )}

      </Select>

      </Form.Item>

      <Form.Item name={['experiment', 'forecasting_horizon']} label="Forecasting horizon">
      <InputNumber min={1} defaultValue={1} onChange={this.onInputNumberChange} />
      </Form.Item>


      <Form.Item name={['experiment', 'mode']} label="Mode" rules={[{ required: false }]}>
      <Select
        showSearch
        style={{ }}
        placeholder="Select the mode"
        optionFilterProp="children"
        onChange={this.onModeChange}
        onFocus={this.onFocus}
        onBlur={this.onBlur}
        onSearch={this.onSearch}
        filterOption={(input, option) =>
          option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
        }
      >


      {this.state.available_modes.map((mode) =>
      <Option value={mode}>{mode}</Option>
      )}

      </Select>
      </Form.Item>


      <Form.Item name={['experiment', 'selected_algos']} label="Selected Algorithms" rules={[{ required: false }]}>
      <Select
      mode="multiple"
      allowClear
      style={{ }}
      placeholder="Please select"
      defaultValue={this.state.selected_algorithms}
      onChange={this.handleMultiSelectChange}
    >
      {this.state.available_algorithms.map((algo) =>
      <Option value={algo}>{algo}</Option>
      )}

    </Select>
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
