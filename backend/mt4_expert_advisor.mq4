//+------------------------------------------------------------------+
//|                                              MT4_Integration.mq4 |
//|                                                                    |
//|                                    AI Trading System Integration |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property link      ""
#property version   "1.00"
#property strict
#property description "AI Trading System - MT4 Integration"

// Archivos de comunicación
string COMMANDS_FILE = "mt4_commands.txt";
string RESPONSES_FILE = "mt4_responses.txt";
string STATUS_FILE = "mt4_status.txt";

// Variables globales
bool isConnected = false;
datetime lastCommandCheck = 0;
int magicNumber = 12345;
string EA_NAME = "AI Trading System";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("AI Trading System - MT4 Integration iniciado");
    
    // No necesitamos crear directorio, usamos el directorio Files por defecto
    Print("Directorio Files disponible para comunicación");
    
    // Escribir estado inicial
    WriteStatus();
    
    // Mostrar mensaje de inicio
    Comment("AI Trading System - Conectado\n",
            "Estado: Activo\n",
            "Cuenta: ", AccountNumber(), "\n",
            "Servidor: ", AccountServer(), "\n",
            "Balance: $", DoubleToString(AccountBalance(), 2), "\n",
            "Última actualización: ", TimeToString(TimeCurrent()));
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("AI Trading System - MT4 Integration detenido");
    Comment(""); // Limpiar comentarios
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Verificar comandos cada segundo
    if(TimeCurrent() - lastCommandCheck >= 1)
    {
        CheckCommands();
        lastCommandCheck = TimeCurrent();
        
        // Actualizar comentario con estado actual
        UpdateComment();
    }
}

//+------------------------------------------------------------------+
//| Actualizar comentario en el gráfico                             |
//+------------------------------------------------------------------+
void UpdateComment()
{
    string status = isConnected ? "Conectado" : "Desconectado";
    string statusIcon = isConnected ? "🟢" : "🔴";
    
    Comment("AI Trading System - ", EA_NAME, "\n",
            "Estado: ", statusIcon, " ", status, "\n",
            "Cuenta: ", AccountNumber(), "\n",
            "Servidor: ", AccountServer(), "\n",
            "Balance: $", DoubleToString(AccountBalance(), 2), "\n",
            "Equity: $", DoubleToString(AccountEquity(), 2), "\n",
            "Símbolo: ", Symbol(), " @ ", DoubleToString(Bid, Digits), "\n",
            "Última actualización: ", TimeToString(TimeCurrent()), "\n",
            "Magic Number: ", magicNumber);
}

//+------------------------------------------------------------------+
//| Verificar comandos desde la aplicación                          |
//+------------------------------------------------------------------+
void CheckCommands()
{
    if(!FileIsExist(COMMANDS_FILE))
        return;
        
    int handle = FileOpen(COMMANDS_FILE, FILE_READ|FILE_TXT);
    if(handle == INVALID_HANDLE)
        return;
        
    string content = "";
    while(!FileIsEnding(handle))
    {
        content += FileReadString(handle);
    }
    FileClose(handle);
    
    // Eliminar archivo de comandos
    FileDelete(COMMANDS_FILE);
    
    if(content != "")
    {
        ProcessCommand(content);
    }
}

//+------------------------------------------------------------------+
//| Procesar comando recibido                                       |
//+------------------------------------------------------------------+
void ProcessCommand(string commandJson)
{
    // Parsear JSON (simplificado)
    string commandType = "";
    string data = "";
    
    Print("Comando JSON completo: ", commandJson);
    
    // Extraer tipo de comando - método corregido
    Print("Buscando 'type' en: ", commandJson);
    
    int typeIndex = StringFind(commandJson, "type");
    Print("Posición de 'type': ", typeIndex);
    
    if(typeIndex >= 0)
    {
        // Buscar después de "type":
        int colonIndex = StringFind(commandJson, ":", typeIndex);
        Print("Posición de ':': ", colonIndex);
        
        if(colonIndex >= 0)
        {
            // Buscar la primera comilla después de ":"
            int quoteIndex = StringFind(commandJson, "\"", colonIndex);
            Print("Posición de primera comilla: ", quoteIndex);
            
            if(quoteIndex >= 0)
            {
                // Buscar la segunda comilla (final del valor)
                int endQuoteIndex = StringFind(commandJson, "\"", quoteIndex + 1);
                Print("Posición de segunda comilla: ", endQuoteIndex);
                
                if(endQuoteIndex >= 0)
                {
                    commandType = StringSubstr(commandJson, quoteIndex + 1, endQuoteIndex - quoteIndex - 1);
                    Print("Tipo extraído: '", commandType, "'");
                }
                else
                {
                    Print("Error: No se encontró el final del valor");
                }
            }
            else
            {
                Print("Error: No se encontró la primera comilla");
            }
        }
        else
        {
            Print("Error: No se encontró ':' después de 'type'");
        }
    }
    else
    {
        Print("Error: No se encontró 'type' en el JSON");
    }
    
    Print("Comando recibido: ", commandType);
    
    if(commandType == "connect")
    {
        Print("Procesando comando CONNECT");
        HandleConnect();
    }
    else if(commandType == "disconnect")
    {
        Print("Procesando comando DISCONNECT");
        HandleDisconnect();
    }
    else if(commandType == "place_order")
    {
        Print("Procesando comando PLACE_ORDER");
        HandlePlaceOrder(commandJson);
    }
    else
    {
        Print("Comando desconocido: ", commandType);
    }
}

//+------------------------------------------------------------------+
//| Manejar comando de conexión                                     |
//+------------------------------------------------------------------+
void HandleConnect()
{
    isConnected = true;
    
    string response = "{";
    response += "\"status\":\"connected\",";
    response += "\"account\":\"" + AccountNumber() + "\",";
    response += "\"server\":\"" + AccountServer() + "\",";
    response += "\"balance\":" + DoubleToString(AccountBalance(), 2) + ",";
    response += "\"equity\":" + DoubleToString(AccountEquity(), 2);
    response += "}";
    
    WriteResponse(response);
    WriteStatus();
    UpdateComment();
    
    Print("Conectado a AI Trading System");
}

//+------------------------------------------------------------------+
//| Manejar comando de desconexión                                  |
//+------------------------------------------------------------------+
void HandleDisconnect()
{
    isConnected = false;
    
    string response = "{\"status\":\"disconnected\"}";
    WriteResponse(response);
    WriteStatus();
    UpdateComment();
    
    Print("Desconectado de AI Trading System");
}

//+------------------------------------------------------------------+
//| Manejar comando de orden                                        |
//+------------------------------------------------------------------+
void HandlePlaceOrder(string commandJson)
{
    // Extraer datos de la orden (simplificado)
    string symbol = "";
    int cmd = OP_BUY;
    double volume = 0.1;
    double price = 0;
    double sl = 0;
    double tp = 0;
    
    // Parsear datos básicos
    if(StringFind(commandJson, "\"symbol\":\"") >= 0)
    {
        int startIndex = StringFind(commandJson, "\"symbol\":\"") + 10;
        int endIndex = StringFind(commandJson, "\"", startIndex);
        if(endIndex >= 0)
        {
            symbol = StringSubstr(commandJson, startIndex, endIndex - startIndex);
        }
    }
    
    if(StringFind(commandJson, "\"type\":\"sell\"") >= 0)
    {
        cmd = OP_SELL;
    }
    
    // Usar símbolo actual si no se especifica
    if(symbol == "")
        symbol = Symbol();
    
    // Colocar orden
    int ticket = OrderSend(symbol, cmd, volume, price, 3, sl, tp, "AI Trading", magicNumber, 0, clrBlue);
    
    string response = "{";
    if(ticket > 0)
    {
        response += "\"status\":\"success\",";
        response += "\"ticket\":" + ticket + ",";
        response += "\"message\":\"Orden colocada exitosamente\"";
        Print("Orden colocada exitosamente - Ticket: ", ticket);
    }
    else
    {
        response += "\"status\":\"error\",";
        response += "\"error\":" + GetLastError() + ",";
        response += "\"message\":\"Error al colocar orden\"";
        Print("Error al colocar orden - Error: ", GetLastError());
    }
    response += "}";
    
    WriteResponse(response);
}

//+------------------------------------------------------------------+
//| Escribir respuesta                                              |
//+------------------------------------------------------------------+
void WriteResponse(string response)
{
    int handle = FileOpen(RESPONSES_FILE, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, response);
        FileClose(handle);
    }
}

//+------------------------------------------------------------------+
//| Escribir estado                                                 |
//+------------------------------------------------------------------+
void WriteStatus()
{
    string status = "{";
    status += "\"connected\":" + (isConnected ? "true" : "false") + ",";
    status += "\"account\":\"" + AccountNumber() + "\",";
    status += "\"server\":\"" + AccountServer() + "\",";
    status += "\"balance\":" + DoubleToString(AccountBalance(), 2) + ",";
    status += "\"equity\":" + DoubleToString(AccountEquity(), 2) + ",";
    status += "\"symbol\":\"" + Symbol() + "\",";
    status += "\"bid\":" + DoubleToString(Bid, Digits) + ",";
    status += "\"ask\":" + DoubleToString(Ask, Digits);
    status += "}";
    
    int handle = FileOpen(STATUS_FILE, FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        FileWriteString(handle, status);
        FileClose(handle);
    }
} 