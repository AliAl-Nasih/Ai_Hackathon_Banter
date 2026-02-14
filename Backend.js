// server.js
import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());

app.post("/debate", async (req, res) => {
  const {topic, userMessage, role} = req.body; // role: 'user' or 'judge', etc.

  // Simple prompt: instruct the LLM to debate against the user
  const prompt = `You are an argumentative debate partner. Topic: ${topic}
User says: "${userMessage}"
Respond with one clear argument or rebuttal in 2-4 sentences. Be concise and give reasoning.`;

  try {
    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini", // example — replace with a model you have access to
        messages: [{role: "system", content: "You are a helpful debate partner."}, {role: "user", content: prompt}],
        max_tokens: 200
      })
    });
    const data = await openaiRes.json();
    const aiText = data.choices?.[0]?.message?.content ?? "Sorry, I couldn't generate a reply.";
    res.json({reply: aiText});
  } catch (err) {
    console.error(err);
    res.status(500).json({error: "backend error"});
  }
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));