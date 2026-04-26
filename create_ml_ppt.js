
const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Deep Agent';
pres.title = 'Machine Learning Introduction';

// Color Palette (Midnight Executive)
const navy = '1E2761';
const iceBlue = 'CADCFC';
const white = 'FFFFFF';

// Title Slide
let slide1 = pres.addSlide();
slide1.background = { fill: navy };
slide1.addText("Machine Learning: An Introduction", {
  x: 0.5, y: 2, w: 9, h: 1.5,
  fontSize: 48, color: white, align: "center", valign: "middle", bold: true
});
slide1.addText("Presented by Deep Agent", {
  x: 0.5, y: 4, w: 9, h: 0.5,
  fontSize: 24, color: iceBlue, align: "center", valign: "middle"
});

// What is Machine Learning?
let slide2 = pres.addSlide();
slide2.background = { fill: white };
slide2.addText("What is Machine Learning?", {
  x: 0.5, y: 0.5, w: 9, h: 0.75,
  fontSize: 36, color: navy, bold: true
});
slide2.addText([
  { text: "Machine Learning (ML) is a subset of Artificial Intelligence (AI).", options: { bullet: true, breakLine: true } },
  { text: "It enables systems to learn from data without explicit programming.", options: { bullet: true, breakLine: true } },
  { text: "ML algorithms build a model based on sample data, known as 'training data', to make predictions or decisions without being explicitly programmed to perform the task.", options: { bullet: true } }
], { x: 0.75, y: 1.5, w: 8.5, h: 4, fontSize: 20, color: '363636' });

// Types of Machine Learning
let slide3 = pres.addSlide();
slide3.background = { fill: white };
slide3.addText("Types of Machine Learning", {
  x: 0.5, y: 0.5, w: 9, h: 0.75,
  fontSize: 36, color: navy, bold: true
});
slide3.addText([
  { text: "Supervised Learning:", options: { bullet: true, bold: true, breakLine: true } },
  { text: "  - Learning from labeled data (e.g., classification, regression).", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Unsupervised Learning:", options: { bullet: true, bold: true, breakLine: true } },
  { text: "  - Finding patterns in unlabeled data (e.g., clustering, dimensionality reduction).", options: { bullet: true, indentLevel: 1, breakLine: true } },
  { text: "Reinforcement Learning:", options: { bullet: true, bold: true, breakLine: true } },
  { text: "  - Learning through trial and error with rewards and penalties.", options: { bullet: true, indentLevel: 1 } }
], { x: 0.75, y: 1.5, w: 8.5, h: 4, fontSize: 20, color: '363636' });

// Applications of Machine Learning
let slide4 = pres.addSlide();
slide4.background = { fill: white };
slide4.addText("Applications of Machine Learning", {
  x: 0.5, y: 0.5, w: 9, h: 0.75,
  fontSize: 36, color: navy, bold: true
});
slide4.addText([
  { text: "Image Recognition: Facial recognition, object detection.", options: { bullet: true, breakLine: true } },
  { text: "Natural Language Processing (NLP): Spam detection, sentiment analysis.", options: { bullet: true, breakLine: true } },
  { text: "Recommendation Systems: Product recommendations on e-commerce sites.", options: { bullet: true, breakLine: true } },
  { text: "Healthcare: Disease diagnosis, drug discovery.", options: { bullet: true, breakLine: true } },
  { text: "Finance: Fraud detection, algorithmic trading.", options: { bullet: true } }
], { x: 0.75, y: 1.5, w: 8.5, h: 4, fontSize: 20, color: '363636' });

// Conclusion Slide
let slide5 = pres.addSlide();
slide5.background = { fill: navy };
slide5.addText("Conclusion", {
  x: 0.5, y: 1.5, w: 9, h: 1,
  fontSize: 48, color: white, align: "center", valign: "middle", bold: true
});
slide5.addText("Machine Learning is transforming industries and enabling intelligent systems.", {
  x: 0.5, y: 3, w: 9, h: 0.75,
  fontSize: 28, color: iceBlue, align: "center", valign: "middle"
});
slide5.addText("Explore, learn, and innovate!", {
  x: 0.5, y: 4, w: 9, h: 0.5,
  fontSize: 24, color: iceBlue, align: "center", valign: "middle"
});

pres.writeFile({ fileName: "Machine_Learning_Introduction.pptx" });
