#!/usr/bin/env python3
"""Send VRAM analysis results via email"""

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import os

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "noreply.airesearch@gmail.com"
SENDER_PASSWORD = "kwqfkeogkozsxffu"
RECIPIENT_EMAIL = "tkdgjs0213@gmail.com"  # Change to actual recipient

def send_email():
    # Create message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "[VRAM Analysis] CacheGen vs Native Mode Comparison"
    
    # Read summary
    with open('/tmp/vram_analysis_summary.txt', 'r') as f:
        summary = f.read()
    
    # Read image
    with open('/tmp/vram_breakdown_comparison.png', 'rb') as f:
        img_data = f.read()
    
    # Attach image
    img = MIMEImage(img_data, 'png')
    img.add_header('Content-Disposition', 'attachment', filename='vram_breakdown_comparison.png')
    msg.attach(img)
    
    # Attach summary as HTML
    html_summary = summary.replace('\n', '<br>\n')
    html_content = f"""
    <html>
    <body>
        <h2>VRAM Breakdown Analysis: CacheGen vs Native Mode</h2>
        <p><b>Key Finding: CacheGen uses MORE VRAM than Native mode!</b></p>
        <p>This is because CacheGen requires additional memory buffers for compression operations.</p>
        <hr>
        {html_summary}
        <hr>
        <h3>VRAM Breakdown Graph:</h3>
        <img src="cid:vram_graph" alt="VRAM Breakdown">
    </body>
    </html>
    """
    msg.attach(MIMEText(html_content, 'html'))
    
    # Add image as inline
    img.add_header('Content-ID', '<vram_graph>')
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

if __name__ == "__main__":
    send_email()
