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
      <div>

              <input type="text" name="name"  value={this.state.exp_name} onChange={this.handleChange}/>

          <buttton onClick={this.createExperiment} >submit</buttton>
  
      </div>
    );
   }
}



export default CreateExperiment;
