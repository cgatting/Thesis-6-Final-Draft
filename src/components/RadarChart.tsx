import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip
} from 'recharts';
import { DimensionScores } from '../types';

interface ScoreRadarProps {
  data: DimensionScores;
}

export const ScoreRadar: React.FC<ScoreRadarProps> = ({ data }) => {
  const chartData = Object.entries(data).map(([key, value]) => ({
    subject: key,
    A: value,
    fullMark: 100,
  }));

  return (
    <div className="w-full h-full min-h-[100px]">
      <ResponsiveContainer width="100%" height="100%" minHeight={100}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700, fontFamily: 'Inter, sans-serif' }} 
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name="RefScore"
            dataKey="A"
            stroke="#4361ee"
            strokeWidth={3}
            fill="#4361ee"
            fillOpacity={0.25}
            isAnimationActive={true}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(15, 23, 42, 0.95)', // slate-950
              borderRadius: '12px', 
              border: '1px solid rgba(255, 255, 255, 0.1)', 
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)',
              color: '#f8fafc' // slate-50
            }}
            itemStyle={{ color: '#60a5fa', fontWeight: 'bold', fontFamily: 'Outfit, sans-serif' }} // blue-400
            cursor={{ stroke: 'rgba(255,255,255,0.2)', strokeWidth: 1 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};