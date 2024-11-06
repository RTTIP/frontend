

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './asset.css';

const App = () => {
    const assetData = [
        { asset_id: 1, name: "HQ Server", type: "Server", value: 120000, criticality: "Critical", created_at: "2024-01-10 09:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 2, name: "Accounting Application", type: "Software", value: 35000, criticality: "High", created_at: "2024-02-20 11:30:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 3, name: "Backup Database", type: "Database", value: 85000, criticality: "High", created_at: "2024-03-15 14:45:00", updated_at: "2024-10-12 17:00:00" },
        { asset_id: 4, name: "Firewall - Main Office", type: "Network", value: 45000, criticality: "Medium", created_at: "2024-05-10 08:20:00", updated_at: "2024-10-15 19:00:00" },
        { asset_id: 5, name: "Employee Workstation 1", type: "Computer", value: 1000, criticality: "Low", created_at: "2024-06-01 10:00:00", updated_at: "2024-09-05 15:30:00" },
        { asset_id: 6, name: "HR Management System", type: "Software", value: 50000, criticality: "High", created_at: "2024-02-28 09:30:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 7, name: "Network Switch 1", type: "Network", value: 2000, criticality: "Medium", created_at: "2024-04-20 08:00:00", updated_at: "2024-10-08 16:00:00" },
        { asset_id: 8, name: "Web Server - E-commerce", type: "Server", value: 70000, criticality: "Critical", created_at: "2024-01-15 11:20:00", updated_at: "2024-10-18 20:00:00" },
        { asset_id: 9, name: "Database Server - Inventory", type: "Server", value: 90000, criticality: "Critical", created_at: "2024-03-18 13:40:00", updated_at: "2024-10-16 14:00:00" },
        { asset_id: 10, name: "Workstation - Design Team", type: "Computer", value: 3000, criticality: "High", created_at: "2024-06-05 09:45:00", updated_at: "2024-09-10 13:30:00" },
        { asset_id: 11, name: "CCTV Camera - Office", type: "Security", value: 500, criticality: "Medium", created_at: "2024-07-15 07:30:00", updated_at: "2024-09-25 15:15:00" },
        { asset_id: 12, name: "Router - Sales Department", type: "Network", value: 1500, criticality: "Medium", created_at: "2024-03-10 10:15:00", updated_at: "2024-10-13 17:30:00" },
        { asset_id: 13, name: "Project Management Tool", type: "Software", value: 25000, criticality: "High", created_at: "2024-05-22 12:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 14, name: "Main Office Generator", type: "Power Supply", value: 40000, criticality: "High", created_at: "2024-01-05 09:00:00", updated_at: "2024-09-20 16:00:00" },
        { asset_id: 15, name: "Laptop - CEO", type: "Computer", value: 1500, criticality: "Critical", created_at: "2024-08-10 10:00:00", updated_at: "2024-09-30 18:45:00" },
        { asset_id: 16, name: "Printer - Admin Office", type: "Peripheral", value: 400, criticality: "Low", created_at: "2024-06-15 08:30:00", updated_at: "2024-08-05 14:30:00" },
        { asset_id: 17, name: "VPN Gateway", type: "Network", value: 6000, criticality: "High", created_at: "2024-02-18 14:10:00", updated_at: "2024-10-11 19:30:00" },
        { asset_id: 18, name: "Sales CRM Software", type: "Software", value: 40000, criticality: "High", created_at: "2024-03-25 09:50:00", updated_at: "2024-10-15 16:15:00" },
        { asset_id: 19, name: "Data Analytics Server", type: "Server", value: 95000, criticality: "Critical", created_at: "2024-01-30 12:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 20, name: "Access Control System", type: "Security", value: 1200, criticality: "Medium", created_at: "2024-04-15 09:30:00", updated_at: "2024-09-10 10:30:00" },
    ];
    

  const [searchQuery, setSearchQuery] = useState("");
  const [filteredAssets, setFilteredAssets] = useState(assetData);

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleSearch = () => {
    const filtered = assetData.filter(asset =>
      asset.asset_id.toString().includes(searchQuery) ||
      asset.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    setFilteredAssets(filtered);
  };

  return (
    <div className="main-content">
      <h2 className="main-title">Assets</h2>
      <div className="search-container">
        <input 
          type="text" 
          placeholder="Search by asset id or name" 
          className="input" 
          value={searchQuery}
          onChange={handleSearchChange}
        />
        <button className="search-button" onClick={handleSearch}>Search</button>
      </div>
      
      {/* Asset Table */}
      <table className="asset-table">
        <thead>
          <tr>
            <th>asset_id</th>
            <th>name</th>
            <th>type</th>
            <th>Value</th>
            <th>criticality</th>
            <th>created_at</th>
            <th>updated_at</th>
          </tr>
        </thead>
        <tbody>
          {filteredAssets.length > 0 ? (
            filteredAssets.map((asset) => (
              <tr key={asset.asset_id}>
                <td>{asset.asset_id}</td>
                <td>{asset.name}</td>
                <td>{asset.type}</td>
                <td>{asset.value}</td>
                <td>{asset.criticality}</td>
                <td>{asset.created_at}</td>
                <td>{asset.updated_at}</td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="7">No assets found</td>
            </tr>
          )}
        </tbody>
      </table>

      {/* Link to assets */}
      {filteredAssets.length === 0 && (
        <Link to="/asset">
          <button className="get-asset-button">View All Assets</button>
        </Link>
      )}
    </div>
  );
};

export default App;
