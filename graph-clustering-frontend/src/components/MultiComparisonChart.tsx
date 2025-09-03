// components/MultiComparisonChart.tsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/Card';

interface ChartDataPoint {
  jobId: string;
  label: string;
  metrics: {
    hmi: number;
    custom_leaf_metric: number;
    custom_displayed_metric: number;
  };
}

interface MultiComparisonChartProps {
  data: ChartDataPoint[];
}

export const MultiComparisonChart: React.FC<MultiComparisonChartProps> = ({ data }) => {
  // Transform data for chart
  const chartData = data.map(exp => ({
    name: exp.label,
    HMI: (exp.metrics.hmi * 100),
    'Leaf Level': (exp.metrics.custom_leaf_metric * 100),
    'Displayed Level': (exp.metrics.custom_displayed_metric * 100)
  }));

  return (
    <Card>
      <h4 className="text-lg font-semibold mb-4">Similarity Metrics Comparison</h4>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={80}
            fontSize={12}
          />
          <YAxis 
            label={{ value: 'Similarity (%)', angle: -90, position: 'insideLeft' }}
            fontSize={12}
          />
          <Tooltip 
            formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
            labelStyle={{ color: '#374151' }}
          />
          <Legend />
          
          <Bar dataKey="HMI" fill="#3b82f6" name="HMI (Hierarchy Similarity)" />
          <Bar dataKey="Leaf Level" fill="#10b981" name="Leaf Level Metric" />
          <Bar dataKey="Displayed Level" fill="#f59e0b" name="Displayed Level Metric" />
        </BarChart>
      </ResponsiveContainer>
      
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>HMI:</strong> Hierarchical Mutual Information - overall hierarchy similarity</p>
        <p><strong>Leaf Level:</strong> Custom metric for entire leaf level graph</p>
        <p><strong>Displayed Level:</strong> Custom metric for largest community comparison</p>
      </div>
    </Card>
  );
};