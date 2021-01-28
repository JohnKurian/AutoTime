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


import { Card } from 'antd';

import create_experiment from './create_experiment.png'; // Tell webpack this JS file uses this image
import resume_experiment from './resume_experiment.png'; // Tell webpack this JS file uses this image

const { Meta } = Card;


function Home() {
  const [placeholder, setPlaceholder] = useState('Hi');

  useEffect(() => {
    fetch('/hello').then(res => res.json()).then(data => {
      setPlaceholder(data.result);
    });
  }, []);


  return (
    <div style={{'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center'}}>
        <Link to="/create_exp">
          <Card
          hoverable
          style={{ width: 240, 'display': 'flex', 'alignItems': 'center', 'flexDirection': 'column' }}
          cover={<img style={{'display': 'flex', 'width': '85px' }} src={create_experiment} alt="Logo" />}
          >
          <Meta title="Create experiment"  />
        </Card>

       </Link>


       <Link to="/experiments">
          <Card
          hoverable
          style={{ width: 240, 'display': 'flex', 'alignItems': 'center',  'flexDirection': 'column' }}
          cover={<img style={{'display': 'flex', 'width': '85px' }} src={resume_experiment} alt="Logo" />}
          >
          <Meta title="Select experiment"  />
        </Card>

       </Link>


    </div>
  );
}




export default Home;
