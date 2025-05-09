import React from "react";
import { AppBar, Toolbar, Typography, Container, Box, Paper, TextField, IconButton, InputAdornment, Stack } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import { useState } from "react";
import Tooltip from "@mui/material/Tooltip";



function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [inputText, setInputText] = useState("");
  const [messages, setMessages] = useState([
    { sender: "assistant", text: "Hi! I'm your shopping assistant. Ask me anything." },
  ]);

  const handleSend = async () => {
    if (!inputText.trim() && !imagePreview) return;

    setMessages((prev) => [
      ...prev,
      { sender: "user", text: inputText, image: imagePreview },
    ]);
    setIsTyping(true);

    try {
      let data;

      if (imagePreview) {
        const file = await fetch(imagePreview).then((r) => r.blob());
        const formData = new FormData();
        formData.append("file", file);
        formData.append("message", inputText);

        const res = await fetch("http://localhost:8000/api/image-search", {
          method: "POST",
          body: formData,
        });
        data = await res.json();
      } else {
        const res = await fetch("http://localhost:8000/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: inputText }),
        });
        data = await res.json();
      }

      setMessages((prev) => [...prev, { sender: "assistant", text: data.reply }]);

      if (data.products && data.products.length > 0) {
        data.products.forEach((product) => {
          setMessages((prev) => [
            ...prev,
            { sender: "assistant", product },
          ]);
        });
      }
    } catch (error) {
      console.error("Error talking to backend:", error);
    }

    setInputText("");
    setImagePreview(null);
    setIsTyping(false);
  };


  const handleImageUpload = (file) => {
    if (!file) return;
    setUploadedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  };


  const handleImageSearch = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/api/image-search", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setMessages(prev => [...prev, { type: "bot", text: data.reply, products: data.products }]);
  };



  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <AppBar position="static" sx={{ bgcolor: "#2f2f3b" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Palona AI Assistant
          </Typography>
        </Toolbar>
      </AppBar>

      <Container fluid className="flex-grow-1 my-3">
        <Paper elevation={3} sx={{ height: "70vh", overflowY: "auto", p: 2 }}>
          {messages.map((msg, index) => (
            <Box
              key={index}
              display="flex"
              justifyContent={msg.sender === "user" ? "flex-end" : "flex-start"}
              my={1}
            >
              <Paper
                elevation={2}
                sx={{
                  p: 1.5,
                  maxWidth: "60%",
                  bgcolor: msg.sender === "user" ? "#1976d2" : "#f1f1f1",
                  color: msg.sender === "user" ? "#fff" : "#000",
                  borderRadius: 2,
                }}
              >
                {msg.product ? (
                  <>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {msg.product.title}
                    </Typography>
                    <Typography variant="body2">{msg.product.description}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      ${msg.product.price}
                    </Typography>
                    <Box mt={1}>
                      <img
                        src={`http://localhost:8000${msg.product.image}`}
                        alt={msg.product.title}
                        style={{ width: 200, height: "auto", borderRadius: 8 }}
                      />
                    </Box>
                  </>
                ) : msg.image ? (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                    </Typography>
                    <img
                      src={msg.image}
                      alt="Uploaded"
                      style={{ width: "100%", borderRadius: 8 }}
                    />
                    {/* Optional: Show caption if exists */}
                    {msg.text && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {msg.text}
                      </Typography>
                    )}
                  </Box>
                ) : (
                  // Normal text
                  <Typography variant="body1">{msg.text}</Typography>
                )}
              </Paper>

            </Box>
          ))}

          {isTyping && (
            <Box display="flex" justifyContent="flex-start" my={1}>
              <Paper
                elevation={1}
                sx={{
                  p: 1.5,
                  bgcolor: "#f1f1f1",
                  borderRadius: 2,
                  fontStyle: "italic",
                  maxWidth: "40%",
                }}
              >
                <Typography variant="body2">Palona is typing...</Typography>
              </Paper>
            </Box>
          )}

        </Paper>
      </Container>

      <Box sx={{ p: 2, borderTop: "1px solid #eee" }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Upload Image">
            <IconButton component="label" color="primary">
              <input
                hidden
                accept="image/*"
                type="file"
                onChange={(e) => handleImageUpload(e.target.files?.[0])}
              />
              <img
                src="https://img.icons8.com/ios-filled/24/image.png"
                alt="Upload"
                style={{ width: 24, height: 24 }}
              />
            </IconButton>
          </Tooltip>


          {imagePreview && (
            <Box
              mb={1}
              sx={{
                maxWidth: 200,
                position: "relative",
              }}
            >
              <img
                src={imagePreview}
                alt="preview"
                style={{ maxWidth: "100%", borderRadius: 8 }}
              />
              <IconButton
                size="small"
                sx={{ position: "absolute", top: 0, right: 0 }}
                onClick={() => setImagePreview(null)}
              >
                ✕
              </IconButton>
            </Box>
          )}

          <TextField
            fullWidth
            variant="outlined"
            size="small"
            placeholder="Type your message..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={handleSend} color="primary">
                    <SendIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Stack>

      </Box>
    </Box>
  );
}

export default App;