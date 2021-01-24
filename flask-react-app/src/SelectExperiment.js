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
    <div>
        {this.state.experiments && this.state.experiments.map(p => <div onClick={() => this.getExperiment(p.experiment_id)}>{p.experiment_id}, {p.name}</div>)}
    </div>
  )
   }
}




export default SelectExperiment;
