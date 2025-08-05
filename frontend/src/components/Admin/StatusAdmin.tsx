import React from 'react';
import AdminStatus from '../Common/AdminStatus';

const StatusAdmin: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Status Admin</h1>
        <p className="text-gray-400">Panel de administración y estado del sistema</p>
      </div>
      
      <AdminStatus />
    </div>
  );
};

export default StatusAdmin; 