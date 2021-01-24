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


class Experiment extends React.Component {


  constructor() {
    super();
    this.state = {
      originHashtags: '', 
      campaignName: '',
      experiments: []
    };
    this.getRun = this.getRun.bind(this)

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
    <div>
        {this.state.runs && this.state.runs.map(p => <div onClick={() => this.getRun(p.run_id)}>{p.run_id}, {p.name}</div>)}
    </div>
  )
   }
}




export default Experiment;
