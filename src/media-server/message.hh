#ifndef MESSAGE_HH
#define MESSAGE_HH

#include <string>
#include <vector>
#include <optional>
#include <exception>

enum class ClientMessageType {
  Unknown,
  Init,
  Info
};

/* Sent by the client to start streaming */
using ClientInit = struct ClientInitMessage {
  std::optional<std::string> channel;

  int player_width;
  int player_height;
};

/* Sent by the client when playing */
using ClientInfo = struct ClientInfoMessage {

  enum class PlayerEvent {
    Unknown,
    Timer,
    Rebuffer,
    CanPlay,
    AudioAck,
    VideoAck
  };

  enum class PlayerReadyState : int {
    HaveNothing = 0,
    HaveMetadata = 1,
    HaveCurrentData = 2,
    HaveFutureData = 3,
    HaveEnoughData = 4
  };

  PlayerEvent event;

  double video_buffer_len;  /* Length of client's buffer in seconds */
  double audio_buffer_len;

  unsigned int next_video_timestamp;  /* Next segment the client is expecting */
  unsigned int next_audio_timestamp;

  int player_width;
  int player_height;
  PlayerReadyState player_ready_state;

  unsigned int init_id;
};

class BadClientMessageException : public std::exception
{
public:
    explicit BadClientMessageException(const char * message) : msg_(message) {}
    explicit BadClientMessageException(const std::string & message)
      : msg_(message) {}
    virtual ~BadClientMessageException() throw () {}
    virtual const char * what() const throw () { return msg_.c_str(); }

protected:
    std::string msg_;
};

/* Client message format:
 *   <message_type> <json_string>
 */

/* Returns a pair containing the message type and the json payload */
std::pair<ClientMessageType, std::string> unpack_client_msg(const std::string & data);

/* Sent by the client on WS connect to request a channel */
ClientInitMessage parse_client_init_msg(const std::string & data);

/* Sent by the client to inform the server's decisions */
ClientInfoMessage parse_client_info_msg(const std::string & data);

/* Server message format:
 *   [0:2]             message_len (network endian)
 *   [2:2+message_len] json string
 *   [2+message_len:]  data
 */

/* Message sent on initial WS connect */
std::string make_server_hello_msg(const std::vector<std::string> & channels);

/* Message sent to reinitialize the client's sourcebuffer */
std::string make_server_init_msg(
  const std::string & channel,
  const std::string & video_codec,
  const std::string & audio_codec,
  const unsigned int timescale,        /* video timescale */
  const unsigned int init_timestamp,   /* starting timestamp in timescale */
  const unsigned int init_id);

/* Audio segment message, payload contains the init and data */
std::string make_audio_msg(
  const std::string & quality,
  const unsigned int timestamp,           /* pts of segment */
  const unsigned int duration,            /* length of segment in timescale */
  const unsigned int byte_offset,         /* byte offset of fragment */
  const unsigned int total_byte_length);  /* total length of all fragments */

/* Video segment message, payload contains the init and data */
std::string make_video_msg(
  const std::string & quality,
  const unsigned int timestamp,           /* see audio */
  const unsigned int duration,
  const unsigned int byte_offset,
  const unsigned int total_byte_length);

#endif /* MESSAGE_HH */
