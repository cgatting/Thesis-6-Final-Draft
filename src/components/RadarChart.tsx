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
        <RadarChart cx="50%" cy="50%" outerRadius="50%" data={chartData}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: '#EDE0C0', fontSize: 12, fontWeight: 700, fontFamily: 'Inter, sans-serif' }} 
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name="RefScore"
            dataKey="A"
            stroke="#DB9E5C"
            strokeWidth={3}
            fill="#DB9E5C"
            fillOpacity={0.25}
            isAnimationActive={true}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#000000', 
              borderRadius: '12px', 
              border: '1px solid rgba(58, 82, 61, 0.5)', 
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
              color: '#EDE0C0' 
            }}
            itemStyle={{ color: '#DB9E5C', fontWeight: 'bold', fontFamily: 'Outfit, sans-serif' }} 
            cursor={{ stroke: 'rgba(237, 224, 192, 0.2)', strokeWidth: 1 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};