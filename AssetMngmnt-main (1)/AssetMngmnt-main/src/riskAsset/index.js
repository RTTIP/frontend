import React from 'react';
import './index.css';
import { Link } from 'react-router-dom'

const RiskAssets = () => {
    // Sample risk asset data
    const riskData = [
      { risk_id: 'r1', asset_id: 'a1', risk_score: 85.5, risk_description: "Potential security vulnerability detected", threat_level: "High", last_evaluation: "2024-10-20 14:30:00" },
      { risk_id: 'r2', asset_id: 'a2', risk_score: 60.0, risk_description: "Outdated software version", threat_level: "Medium", last_evaluation: "2024-10-18 11:20:00" },
      { risk_id: 'r3', asset_id: 'a3', risk_score: 30.5, risk_description: "Unstable network connection", threat_level: "Low", last_evaluation: "2024-10-15 09:15:00" },
      { risk_id: 'r4', asset_id: 'a1', risk_score: 95.2, risk_description: "Critical data exposure risk", threat_level: "High", last_evaluation: "2024-10-25 16:45:00" },
    ];
  
    return (
      <Link to='/riskAsset' style={{ textDecoration: 'none'}} >
        <div className="main-content">
        <h2 className="main-title">Risk Assets</h2>
        <div className="search-container">
          <input 
            type="text" 
            placeholder="Search by asset id or name" 
            className="input" 
          />
          <button className="search-button">Search</button>
        </div>
        
        {/* Risk Assets Table */}
        <table className="risk-table">
          <thead>
            <tr>
              <th>risk_id</th>
              <th>asset_id</th>
              <th>risk_score</th>
              <th>risk_description</th>
              <th>threat_level</th>
              <th>last_evaluation</th>
            </tr>
          </thead>
          <tbody>
            {riskData.map((risk) => (
              <tr key={risk.risk_id}>
                <td>{risk.risk_id}</td>
                <td>{risk.asset_id}</td>
                <td>{risk.risk_score}</td>
                <td>{risk.risk_description}</td>
                <td>{risk.threat_level}</td>
                <td>{risk.last_evaluation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </Link>
    );
  };
  
  
export default RiskAssets;
