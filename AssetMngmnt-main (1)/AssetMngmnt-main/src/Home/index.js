// import React, { Component } from 'react';
// import { Navigate } from 'react-router-dom';
// import './index.css';



// class Index extends Component {
//   constructor(props) {
//     super(props);
//     this.state = {
//       searchQuery: '',
//       foundAsset: null, // Will hold the found asset or null if not found
//       assets: [
//         { asset_id: 1, name: "HQ Server", description: "Primary server for headquarters", type: "Server", value: 120000, criticality: "Critical", created_at: "2024-01-10 09:00:00", updated_at: "2024-10-10 18:00:00" },
//         { asset_id: 2, name: "Accounting Application", description: "Software for managing finances", type: "Software", value: 35000, criticality: "High", created_at: "2024-02-20 11:30:00", updated_at: "2024-10-10 18:00:00" },
//         { asset_id: 3, name: "Backup Database", description: "Database for backup storage", type: "Database", value: 85000, criticality: "High", created_at: "2024-03-15 14:45:00", updated_at: "2024-10-12 17:00:00" },
//         { asset_id: 4, name: "Firewall - Main Office", description: "Firewall for network security", type: "Network", value: 45000, criticality: "Medium", created_at: "2024-05-10 08:20:00", updated_at: "2024-10-15 19:00:00" },
//         { asset_id: 5, name: "Employee Workstation 1", description: "Workstation assigned to employee", type: "Computer", value: 1000, criticality: "Low", created_at: "2024-06-01 10:00:00", updated_at: "2024-09-05 15:30:00" },
//         { asset_id: 6, name: "HR Management System", description: "Software for HR management", type: "Software", value: 50000, criticality: "High", created_at: "2024-02-28 09:30:00", updated_at: "2024-10-10 18:00:00" },
//         { asset_id: 7, name: "Network Switch 1", description: "Switch for network connectivity", type: "Network", value: 2000, criticality: "Medium", created_at: "2024-04-20 08:00:00", updated_at: "2024-10-08 16:00:00" },
//         { asset_id: 8, name: "Web Server - E-commerce", description: "Server for e-commerce website", type: "Server", value: 70000, criticality: "Critical", created_at: "2024-01-15 11:20:00", updated_at: "2024-10-18 20:00:00" },
//         { asset_id: 9, name: "Database Server - Inventory", description: "Server hosting inventory database", type: "Server", value: 90000, criticality: "Critical", created_at: "2024-03-18 13:40:00", updated_at: "2024-10-16 14:00:00" },
//         { asset_id: 10, name: "Workstation - Design Team", description: "Workstation for design team", type: "Computer", value: 3000, criticality: "High", created_at: "2024-06-05 09:45:00", updated_at: "2024-09-10 13:30:00" },
//         { asset_id: 11, name: "CCTV Camera - Office", description: "Security camera for office surveillance", type: "Security", value: 500, criticality: "Medium", created_at: "2024-07-15 07:30:00", updated_at: "2024-09-25 15:15:00" },
//         { asset_id: 12, name: "Router - Sales Department", description: "Router for sales department network", type: "Network", value: 1500, criticality: "Medium", created_at: "2024-03-10 10:15:00", updated_at: "2024-10-13 17:30:00" },
//         { asset_id: 13, name: "Project Management Tool", description: "Tool for managing projects", type: "Software", value: 25000, criticality: "High", created_at: "2024-05-22 12:00:00", updated_at: "2024-10-10 18:00:00" },
//         { asset_id: 14, name: "Main Office Generator", description: "Backup generator for office power", type: "Power Supply", value: 40000, criticality: "High", created_at: "2024-01-05 09:00:00", updated_at: "2024-09-20 16:00:00" },
//         { asset_id: 15, name: "Laptop - CEO", description: "Laptop assigned to CEO", type: "Computer", value: 1500, criticality: "Critical", created_at: "2024-08-10 10:00:00", updated_at: "2024-09-30 18:45:00" },
//         { asset_id: 16, name: "Printer - Admin Office", description: "Printer for administration office", type: "Peripheral", value: 400, criticality: "Low", created_at: "2024-06-15 08:30:00", updated_at: "2024-08-05 14:30:00" },
//         { asset_id: 17, name: "VPN Gateway", description: "Gateway for VPN connections", type: "Network", value: 6000, criticality: "High", created_at: "2024-02-18 14:10:00", updated_at: "2024-10-11 19:30:00" },
//         { asset_id: 18, name: "Sales CRM Software", description: "Software for managing sales CRM", type: "Software", value: 40000, criticality: "High", created_at: "2024-03-25 09:50:00", updated_at: "2024-10-15 16:15:00" },
//         { asset_id: 19, name: "Data Analytics Server", description: "Server for data analytics tasks", type: "Server", value: 95000, criticality: "Critical", created_at: "2024-01-30 12:00:00", updated_at: "2024-10-10 18:00:00" },
//         { asset_id: 20, name: "Access Control System", description: "System for access control", type: "Security", value: 1200, criticality: "Medium", created_at: "2024-04-15 09:30:00", updated_at: "2024-09-10 10:30:00" }
//     ],
//       assetNotFound: false, // New state to track if no asset is found
//       isRedirect: false, // Added to control navigation
//     };
//   }

//   handleSearchChange = (event) => {
//     this.setState({ searchQuery: event.target.value });
//   };

//   handleSearch = () => {
//     const { searchQuery, assets } = this.state;
//     const foundAsset = assets.find(
//       asset => asset.asset_id === searchQuery || asset.name.toLowerCase().includes(searchQuery.toLowerCase())
//     );

//     if (foundAsset) {
//       this.setState({ foundAsset, assetNotFound: false }); // Show the asset details
//     } else {
//       this.setState({ foundAsset: null, assetNotFound: true }); // Show "Asset Not Found" message
//     }
//   };

//   redirecting = () => {
//     this.setState({ isRedirect: true }); // Set redirect flag to true
//   }

//   render() {
//     const { searchQuery, foundAsset, assetNotFound, isRedirect } = this.state;

//     if (isRedirect) {
//       // Trigger redirection if flag is set
//       return <Navigate to="/asset" replace />;
//     }

//     return (
//       <div className="main-content">
//         <h2 className="main-title">Search for an asset</h2>
//         <div className="search-container">
//           <input
//             type="text"
//             placeholder="Search by asset id or name"
//             className="input"
//             value={searchQuery}
//             onChange={this.handleSearchChange}
//           />
//           <button className="search-button" onClick={this.handleSearch}>
//             Search
//           </button>
//         </div>

//         {foundAsset ? (
//           // If asset is found, display its details
//           <div className="asset-details">
//             <h3>Asset Details</h3>
//             <p><strong>Asset ID:</strong> {foundAsset.asset_id}</p>
//             <p><strong>Name:</strong> {foundAsset.name}</p>
//             <p><strong>Description:</strong> {foundAsset.description}</p>
//           </div>
//         ) : assetNotFound ? (
//           // If no asset is found, show "Asset Not Found" message
//           <div className="asset-not-found">
//             <p>No asset found with that ID or name.</p>
//             <button onClick={this.redirecting} className="get-asset-button">View All Assets</button>
//           </div>
//         ) : (
//           // Optional placeholder if no search has been performed
//           <p>Enter an asset ID or name to search.</p>
//         )}
//       </div>
//     );
//   }
// }

// export default Index;


import React, { Component } from 'react';
import { Navigate } from 'react-router-dom';
import './index.css';

class Index extends Component {
  constructor(props) {
    super(props);
    this.state = {
      searchQuery: '',
      foundAssets: [], // To hold filtered assets
      assets: [
        { asset_id: 1, name: "HQ Server", description: "Primary server for headquarters", type: "Server", value: 120000, criticality: "Critical", created_at: "2024-01-10 09:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 2, name: "Accounting Application", description: "Software for managing finances", type: "Software", value: 35000, criticality: "High", created_at: "2024-02-20 11:30:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 3, name: "Backup Database", description: "Database for backup storage", type: "Database", value: 85000, criticality: "High", created_at: "2024-03-15 14:45:00", updated_at: "2024-10-12 17:00:00" },
        { asset_id: 4, name: "Firewall - Main Office", description: "Firewall for network security", type: "Network", value: 45000, criticality: "Medium", created_at: "2024-05-10 08:20:00", updated_at: "2024-10-15 19:00:00" },
        { asset_id: 5, name: "Employee Workstation 1", description: "Workstation assigned to employee", type: "Computer", value: 1000, criticality: "Low", created_at: "2024-06-01 10:00:00", updated_at: "2024-09-05 15:30:00" },
        { asset_id: 6, name: "HR Management System", description: "Software for HR management", type: "Software", value: 50000, criticality: "High", created_at: "2024-02-28 09:30:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 7, name: "Network Switch 1", description: "Switch for network connectivity", type: "Network", value: 2000, criticality: "Medium", created_at: "2024-04-20 08:00:00", updated_at: "2024-10-08 16:00:00" },
        { asset_id: 8, name: "Web Server - E-commerce", description: "Server for e-commerce website", type: "Server", value: 70000, criticality: "Critical", created_at: "2024-01-15 11:20:00", updated_at: "2024-10-18 20:00:00" },
        { asset_id: 9, name: "Database Server - Inventory", description: "Server hosting inventory database", type: "Server", value: 90000, criticality: "Critical", created_at: "2024-03-18 13:40:00", updated_at: "2024-10-16 14:00:00" },
        { asset_id: 10, name: "Workstation - Design Team", description: "Workstation for design team", type: "Computer", value: 3000, criticality: "High", created_at: "2024-06-05 09:45:00", updated_at: "2024-09-10 13:30:00" },
        { asset_id: 11, name: "CCTV Camera - Office", description: "Security camera for office surveillance", type: "Security", value: 500, criticality: "Medium", created_at: "2024-07-15 07:30:00", updated_at: "2024-09-25 15:15:00" },
        { asset_id: 12, name: "Router - Sales Department", description: "Router for sales department network", type: "Network", value: 1500, criticality: "Medium", created_at: "2024-03-10 10:15:00", updated_at: "2024-10-13 17:30:00" },
        { asset_id: 13, name: "Project Management Tool", description: "Tool for managing projects", type: "Software", value: 25000, criticality: "High", created_at: "2024-05-22 12:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 14, name: "Main Office Generator", description: "Backup generator for office power", type: "Power Supply", value: 40000, criticality: "High", created_at: "2024-01-05 09:00:00", updated_at: "2024-09-20 16:00:00" },
        { asset_id: 15, name: "Laptop - CEO", description: "Laptop assigned to CEO", type: "Computer", value: 1500, criticality: "Critical", created_at: "2024-08-10 10:00:00", updated_at: "2024-09-30 18:45:00" },
        { asset_id: 16, name: "Printer - Admin Office", description: "Printer for administration office", type: "Peripheral", value: 400, criticality: "Low", created_at: "2024-06-15 08:30:00", updated_at: "2024-08-05 14:30:00" },
        { asset_id: 17, name: "VPN Gateway", description: "Gateway for VPN connections", type: "Network", value: 6000, criticality: "High", created_at: "2024-02-18 14:10:00", updated_at: "2024-10-11 19:30:00" },
        { asset_id: 18, name: "Sales CRM Software", description: "Software for managing sales CRM", type: "Software", value: 40000, criticality: "High", created_at: "2024-03-25 09:50:00", updated_at: "2024-10-15 16:15:00" },
        { asset_id: 19, name: "Data Analytics Server", description: "Server for data analytics tasks", type: "Server", value: 95000, criticality: "Critical", created_at: "2024-01-30 12:00:00", updated_at: "2024-10-10 18:00:00" },
        { asset_id: 20, name: "Access Control System", description: "System for access control", type: "Security", value: 1200, criticality: "Medium", created_at: "2024-04-15 09:30:00", updated_at: "2024-09-10 10:30:00" }
      ],
      assetNotFound: false, // New state to track if no asset is found
      isRedirect: false, // Added to control navigation
    };
  }

  handleSearchChange = (event) => {
    this.setState({ searchQuery: event.target.value });
  };

  handleSearch = () => {
    const { searchQuery, assets } = this.state;
    const foundAssets = assets.filter(
      asset => asset.asset_id.toString().includes(searchQuery) || asset.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (foundAssets.length > 0) {
      this.setState({ foundAssets, assetNotFound: false }); // Show filtered assets
    } else {
      this.setState({ foundAssets: [], assetNotFound: true }); // No assets found
    }
  };

  redirecting = () => {
    this.setState({ isRedirect: true }); // Set redirect flag to true
  }

  render() {
    const { searchQuery, foundAssets, assetNotFound, isRedirect } = this.state;

    if (isRedirect) {
      return <Navigate to="/asset" replace />;
    }

    return (
      <div className="main-content">
        <h2 className="main-title">Search for an asset</h2>
        <div className="search-container">
          <input
            type="text"
            placeholder="Search by asset id or name"
            className="input"
            value={searchQuery}
            onChange={this.handleSearchChange}
          />
          <button className="search-button" onClick={this.handleSearch}>
            Search
          </button>
        </div>

        {foundAssets.length > 0 ? (
          // If assets are found, display them in a table
          <table className="asset-table">
            <thead>
              <tr>
                <th>Asset ID</th>
                <th>Name</th>
                <th>Description</th>
                <th>Type</th>
                <th>Value</th>
                <th>Criticality</th>
                <th>Created At</th>
                <th>Updated At</th>
              </tr>
            </thead>
            <tbody>
              {foundAssets.map(asset => (
                <tr key={asset.asset_id}>
                  <td>{asset.asset_id}</td>
                  <td>{asset.name}</td>
                  <td>{asset.description}</td>
                  <td>{asset.type}</td>
                  <td>{asset.value}</td>
                  <td>{asset.criticality}</td>
                  <td>{asset.created_at}</td>
                  <td>{asset.updated_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          assetNotFound && <p>No asset found matching the search criteria.</p>
        )}
      </div>
    );
  }
}

export default Index;
