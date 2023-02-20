---
layout: article
title: 计算机网络基础
key: 100012
tags: 计算机网络
category: blog
date: 2020-10-16 00:00:00 +08:00
mermaid: true
---

# HTTP实现
 HTTP实现包括server端和client端，其中server端用于处理client端发送的请求

## server.h
 HTTP需要设置server最大缓存大小、允许的最大连接数量、当前client传递的套接字、以及client传送的请求信息、客户端地址的内存长度，以及发送信息给client


<!--more-->

  ```c++
  #include <WinSock2.h>
  #include <WS2tcpip.h>
  #pragma once
  // 最大传输字符数量 10M
  #if !MAX_CONNECT_CHAR_BUFFER
  #define MAX_CONNECT_CHAR_BUFFER 10485760  
  #endif

  class server{
    public:

    // c:ipv4 port:请求端口
      server(const char* c, u_short port);
      server(const char* c, u_short port, int type);

    // 监听 允许监听的最大num数量
    void listen(int num);

    ~server();

    // 套接字 WinSock2.h
    SOCKET s;
    
    // 当前客户端的请求信息 WS2tcpip.h
    SOCKADDR_IN serverIn;

    // 链接的客户端地址信息的内存长度
    int nowClientAddrLen;

    // 当前请求的缓存内容
    char buffer[MAX_CONNECT_CHAR_BUFFER];

    // 往客户端发送字符串
    BOOL send(string s);
    BOOL send(string s, SOCKET s2);

    BOOL allSend(string s);
    int errorInt;
    BOOL waitUDP(SOCKET s, char* bufer, const char* host, int port, int flags=0);
    vector<SOCKET> clientArr;
    vector<int> socketIndex;
    BOOL createThread(int index);
    string nowClientMsg;
  };
  ```

### server.cpp

  ```c++
  #include "server.h"

  server::server(const char* c, u_short port){
    memset(&serverIn, 0, sizeof(serverIn));
    serverIn.sin_family = AF_INET;
    serverIn.sin_port = htons(port);
    serverIn.sin_addr.S_un.S_addr = inet_addr(c);
    s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  };

  server::server(const char* c, u_short port, int type){
    memset(&serverIn, 0, sizeof(serverIn));
    serverIn.sin_family = AF_INET;
    serverIn.sin_port = htons(port);
    serverIn.sin_addr.S_un.S_addr = inet_addr(c);
    s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  };

  void server::listen(int num){

  };

  BOOL server::allSend(string s){
    for(unsigned int i = 0; i< clientArr.size(); i++){
      this.send(s, clientArr[i]);
    }
  };

  BOOL server::send(string s, SOCKET s2){
    errorInt =::send(s2, s.c_str(), s.length() +1 , 0);
    if(errorInt == SOCKET_ERROR){
      return FLASE;
    }else{
      return TRUE;
    }
  };

  BOOL server::send(string s) {
    return this->send(s, nowClient);
  };

  BOOL server::createThread(int index){
    HANDLE ev = WSACreateEvent();
    WSAEventSelect(this->clientArr[index], ev, FD_ACCEPT | FD_CLOSE);
    // 1:数组中时间对象句柄的数目，此处为1
    // &ev 指向一个事件对象句柄数组的指针
    // TRUE：指定等待类型，如果为TRUE，则需要所有事件对象同时有信号，如果为FALSE，需要任意事件对象有信号时返回，返回值指出哪个事件对象造成函数返回
    // INFINITE：指定超时时间间隔，当超时间隔到，函数即返回
    // 一般设置为TRUE
    WSAWaitForMultipleEvents(1, &ev, TRUE, INFINITE, FALSE);
  };

  void server::listen(int num){
    
  }

  server::~server(){

  }

  ```


### oMap.h & oMap.cpp
 用于client的属性封装类
  ```c++
  #pragma once
  template <class key, class value>
  class oMap{
    public:
      oMap();
      ~oMap();
      bool set(key, value);
      bool find(key);
      string toString();
      value get(key);
      value& operator [](key);
      BOOL foreach(void(*) (int, value, key));
    public:
      vector<key> _key;
      vector<value> _value;
  };

  template <class key, class value>
  oMap::oMap(){

  }

  ```

### helpClass.h
 定义了client的static方法，其中实现了get方法


  ```c++
  #pragma once
  class helpClass{

    public:
    helpClass();
    ~helpClass();

  // 拆分字符串
  static vector<string> split(string s, string s2);

  // 获取请求头
  static oMap<string, string> getRequestHead(string request);

  // 获取默认的响应头
  static string getResponseHead(string request, int status, string statusMsg, oMap<string, string>headObj);

  // 获取系统时间
  static string getTime();

  // 获取完整的请求文件地址
  static string getFileUrl(string url,string methodType, oMap<string, string>getMap);

  // 获取文件字符集
  static string helpClass::getFileCharset(string s);

  // 获取minitype
  static string getMINItype(string path);

  // 字符串转码方法
  static LPWSTR user_stringToLPWSTR(string orig);
  static string user_LPWSTRTostring(LPWSTR lpw);
  static LPCSTR user_LPWSTRToLPCSTR(LPWSTR lpw);
  static LPWSTR user_LPCSTRToLPWSTR(LPCSTR lpc);

  // client入口函数
  static void handleRequestLine(SOCKET& s, string request);

  // 处理get请求
  static void helpClass::methodGetHandle(SOCKET& s, string request, oMap<string, string> getMap);

  // 读取文件的方法
  static int readFile(string filepath, string& content);


  };
  namespace METHOD_TYPE {
    const string uHTTP_GET = "GET";
    const string uHTTP_POST = "POST";
    const string uHTTP_OPTIONS = "OPTIONS";
    const string uHTTP_DELETE = "DELETE";
    const string uHTTP_PUT = "PUT";
  };


  ```

### helpClass.cpp

  ```c++
  #include "helpClass.cpp"

  #define DEFAULT_PATH ""

  helpClass::helpClass(){}
  helpClass::~helpClass(){}

  // 字符串转换算法的实现
  LPWSTR helpClass::user_stringToLPWSTR(string orig){
    int leg = orig.length();
    LPWSTR d = malloc(sizeof(LPWSTR)* (leg+1));
    memset(d, 0, sizeof(LPWSTR)* (leg+1));

    // 编码方式、offset、char*、length、LPWSTR、size
    MultiByteToWideChar(CP_ACP, 0, orig.c_str(), lng, d, lng * sizeof(PWSTR));
    return d;
  }
  string helpClass::user_LPWSTRTostring(LPWSTR lpw) {
    int lng = lstrlenW(lpw);
    LPSTR d = (LPSTR)malloc(sizeof(PSTR)*lng);
    memset(d, 0, sizeof(PSTR)*lng);


    WideCharToMultiByte(CP_OEMCP, NULL, lpw, -1, d, sizeof(PSTR)*lng, NULL, FALSE);
    string q = d;
    return q;
  }
  LPCSTR helpClass::user_LPWSTRToLPCSTR(LPWSTR lpw) {
    string s1 = helpClass::user_LPWSTRTostring(lpw);
    return (s1.c_str());
  }


  // 拆分字符串
  vector<string> helpClass::split(string s, string s2){
    
  }

  // 获取请求头
   oMap<string, string> helpClass::getRequestHead(string request);

  // 获取默认的响应头
  static string getResponseHead(string request, int status, string statusMsg, oMap<string, string>headObj);

  // 获取系统时间
  static string getTime();

  // 获取完整的请求文件地址
  static string getFileUrl(string url,string methodType, oMap<string, string>getMap);

  // 获取文件字符集
  static string helpClass::getFileCharset(string s);

  // 获取minitype
  static string getMINItype(string path);

  // 字符串转码方法
  static LPWSTR user_stringToLPWSTR(string orig);
  static string user_LPWSTRTostring(LPWSTR lpw);
  static LPCSTR user_LPWSTRToLPCSTR(LPWSTR lpw);
  static LPWSTR user_LPCSTRToLPWSTR(LPCSTR lpc);

  // client入口函数
  static void handleRequestLine(SOCKET& s, string request);

  // 处理get请求
  static void helpClass::methodGetHandle(SOCKET& s, string request, oMap<string, string> getMap);

  // 读取文件的方法
  static int readFile(string filepath, string& content);


  // 

  ```
