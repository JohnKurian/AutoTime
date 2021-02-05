import React from 'react';
import ReactDOM from 'react-dom';
import CreateExperiment from './CreateExperiment.js';
import App from './App';

import { Layout, Menu, Breadcrumb } from 'antd';
import {
  ExperimentOutlined,
  HomeOutlined ,
  SettingFilled,
  TeamOutlined,
  UserOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
    Redirect
  } from "react-router-dom";

import "./siderdemo.css";
import logo from './abb_white.png'; // Tell webpack this JS file uses this image

const { Header, Content, Footer, Sider } = Layout;
const { SubMenu } = Menu;




class SiderDemo extends React.Component {
  state = {
    collapsed: false,
  };

  onCollapse = collapsed => {
    console.log(collapsed);
    this.setState({ collapsed });
  };

  render() {
    const { collapsed } = this.state;
    return (
        <Router>
      <Layout style={{ height: '100vh' }}>
        <Sider 
        collapsible collapsed={collapsed} 
        onCollapse={this.onCollapse}
        style={{ 
         
      }}
        >
          <div className="logo" />
          <Menu theme="dark" defaultSelectedKeys={['1']} mode="inline">
            <Menu.Item key="1" icon={<HomeOutlined  />}>
            <Link to="/">Home</Link>
            </Menu.Item>
            <Menu.Item key="2" icon={<ExperimentOutlined />}>
            <Link to="/experiments">Experiments</Link>
            </Menu.Item>

            <Menu.Item key="3" icon={<DatabaseOutlined />}>
            <Link to="/datasets">Datasets</Link>
            </Menu.Item>

            <Menu.Item key="4" icon={<SettingFilled />}>
            <Link to="/settings">Settings</Link>
            </Menu.Item>
          </Menu>
        </Sider>
        <Layout className="site-layout">
          <Header className="site-layout-background" style={{ padding: 0, 'display': 'flex', 'background': '#001529', 'height': '52px' }}><img style={{ 'width': '85px', 'alignItems': 'center', 'alignSelf': 'center' }} src={logo} alt="Logo" /></Header>
          <Content style={{ margin: '0 16px', overflow: 'scroll'  }}>
            <Breadcrumb style={{ margin: '16px 0' }}>
            </Breadcrumb>
            <div className="site-layout-background" style={{ padding: 24, minHeight: 360 }}>
                <App/>
            </div>
          </Content>
        </Layout>
      </Layout>
      </Router>
    );
  }
}

export default SiderDemo;