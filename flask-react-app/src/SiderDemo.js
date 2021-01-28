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
} from '@ant-design/icons';

import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
    Redirect
  } from "react-router-dom";

import "./siderdemo.css";
import logo from './abb_logo.png'; // Tell webpack this JS file uses this image

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
      <Layout style={{ minHeight: '100vh' }}>
        <Sider collapsible collapsed={collapsed} onCollapse={this.onCollapse}>
          <div className="logo" />
          <Menu theme="dark" defaultSelectedKeys={['1']} mode="inline">
            <Menu.Item key="1" icon={<HomeOutlined  />}>
            <Link to="/">Home</Link>
            </Menu.Item>
            <Menu.Item key="2" icon={<ExperimentOutlined />}>
            <Link to="/experiments">Experiments</Link>
            </Menu.Item>
            <Menu.Item key="3" icon={<SettingFilled />}>
            <Link to="/settings">Settings</Link>
            </Menu.Item>
          </Menu>
        </Sider>
        <Layout className="site-layout">
          <Header className="site-layout-background" style={{ padding: 0 }}><img style={{ 'width': '85px' }} src={logo} alt="Logo" /></Header>
          <Content style={{ margin: '0 16px' }}>
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