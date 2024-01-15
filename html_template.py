css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #635a47
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 80px;
  max-height: 80px;
  border-radius: 50%;
  object-fit: fit;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #fff;
font-size: 16px;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/736x/78/c5/da/78c5da9e83b843f4f64d08d3d95ae9c4.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/736x/56/0e/5c/560e5ce944d0ae2cb83d4fea78003fb4.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''