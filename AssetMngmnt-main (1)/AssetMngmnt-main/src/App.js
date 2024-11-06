import React, { Component, Fragment } from 'react';
import Home from './Home/index';
import Asset from './asset/asset';
import RiskAssets from './riskAsset';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import './App.css';

class App extends Component {
  assetFn = () => {
    this.props.navigate('/asset');
  };

  riskAssetFn = () => {
    this.props.navigate('/riskAsset');
  };

  render() {
    return (
      <div className="container">
        <Fragment>
          <div className="sidebar">
            <h2 className="sidebar-title">
              <Link to="/" style={{ textDecoration: 'none', color: 'white' }}>
                Asset Management
              </Link>
            </h2>
            <div onClick={this.assetFn} className="menu-item">Assets</div>
            <div onClick={this.riskAssetFn} className="menu-item">Risk Assets</div>
            <div className="menu-item">Incidents</div>
          </div>
          <div className="content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/asset" element={<Asset />} />
              <Route path="/riskAsset" element={<RiskAssets />} />
            </Routes>
          </div>
        </Fragment>
      </div>
    );
  }
}

const AppWithNavigate = (props) => {
  const navigate = useNavigate();
  return <App {...props} navigate={navigate} />;
};

export default AppWithNavigate;
