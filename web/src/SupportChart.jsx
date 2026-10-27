import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

export default function SupportChart({ timeline }) {
  const data = timeline.map((observation) => ({
    round: observation.round_index,
    support: Number(observation.aggregate_support) * 100,
  }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 8, left: -20, bottom: 0 }}>
        <defs><linearGradient id="supportGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#8de2be" stopOpacity={0.26} /><stop offset="100%" stopColor="#8de2be" stopOpacity={0.01} /></linearGradient></defs>
        <CartesianGrid stroke="#dce7e1" strokeDasharray="3 6" vertical={false} />
        <XAxis dataKey="round" tickLine={false} axisLine={false} tick={{ fill: "#7b8d85", fontSize: 11 }} tickFormatter={(round) => `R${round}`} />
        <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: "#7b8d85", fontSize: 10 }} tickFormatter={(value) => `${value}%`} />
        <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #dce7e1", borderRadius: 10, color: "#172321", fontSize: 12 }} formatter={(value) => [`${Number(value).toFixed(2)}%`, "Aggregate support"]} labelFormatter={(round) => `Round ${round}`} />
        <Area type="monotone" dataKey="support" stroke="#23946d" strokeWidth={2.5} fill="url(#supportGradient)" activeDot={{ r: 5, fill: "#23946d", stroke: "#ffffff", strokeWidth: 2 }} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
