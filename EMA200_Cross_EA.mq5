//+------------------------------------------------------------------+
//|                                             EMA200_Cross_EA.mq5  |
//|  EA de cruces de EMA 200 (pensado para XAUUSD, sirve en          |
//|  cualquier simbolo/timeframe).                                   |
//|                                                                  |
//|  Logica:                                                         |
//|    - COMPRA cuando la vela anterior CIERRA por ENCIMA de la      |
//|      EMA 200 habiendo estado antes por debajo (cruce alcista).   |
//|    - VENDE cuando la vela anterior CIERRA por DEBAJO de la       |
//|      EMA 200 habiendo estado antes por encima (cruce bajista).   |
//|    - Usa velas CERRADAS (no repinta ni entra a mitad de vela).   |
//|    - Una sola posicion a la vez; en cruce contrario cierra y     |
//|      (si asi se configura) abre en la nueva direccion.           |
//|    - SL y TP en puntos + trailing stop opcional.                 |
//|                                                                  |
//|  Nota XAUUSD: 1 punto = 0.01 USD en la mayoria de brokers.       |
//|  Ej.: SL de 3000 puntos = 30.00 USD de recorrido.                |
//|                                                                  |
//|  ADVERTENCIA: herramienta educativa. Prueba SIEMPRE en el        |
//|  probador de estrategias y en cuenta demo antes de usar dinero   |
//|  real. Ningun EA garantiza ganancias.                            |
//+------------------------------------------------------------------+
#property copyright  "Uso educativo"
#property version    "1.00"
#property description "Cruce de EMA 200 con velas cerradas, SL/TP en puntos y trailing opcional"

#include <Trade\Trade.mqh>

//--- parametros de entrada
input int             InpEmaPeriod        = 200;            // Periodo de la EMA
input ENUM_TIMEFRAMES InpTimeframe        = PERIOD_CURRENT; // Timeframe de la senal
input double          InpLots             = 0.10;           // Lote fijo
input int             InpStopLossPoints   = 3000;           // Stop Loss en puntos (0 = sin SL)
input int             InpTakeProfitPoints = 6000;           // Take Profit en puntos (0 = sin TP)
input int             InpTrailingPoints   = 0;              // Trailing stop en puntos (0 = desactivado)
input int             InpTrailingStepPts  = 50;             // Paso minimo del trailing en puntos
input bool            InpCloseOnOpposite  = true;           // Cerrar posicion en cruce contrario
input bool            InpOpenOnOpposite   = true;           // Abrir en la nueva direccion tras cerrar
input int             InpSlippagePoints   = 20;             // Desviacion maxima en puntos
input ulong           InpMagic            = 200200;         // Numero magico (identifica al EA)
input string          InpTradeComment     = "EMA200 cross"; // Comentario de las operaciones

//--- globales
CTrade   g_trade;
int      g_emaHandle   = INVALID_HANDLE;
datetime g_lastBarTime = 0;

//+------------------------------------------------------------------+
//| Inicializacion                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_emaHandle = iMA(_Symbol, InpTimeframe, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(g_emaHandle == INVALID_HANDLE)
     {
      Print("Error: no pude crear el indicador EMA (", GetLastError(), ")");
      return INIT_FAILED;
     }
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   Print("EMA200 Cross EA iniciado en ", _Symbol, " ", EnumToString(InpTimeframe),
         " | EMA ", InpEmaPeriod, " | lote ", DoubleToString(InpLots, 2));
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Limpieza                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_emaHandle != INVALID_HANDLE)
      IndicatorRelease(g_emaHandle);
  }

//+------------------------------------------------------------------+
//| Tick principal                                                   |
//+------------------------------------------------------------------+
void OnTick()
  {
   // el trailing se gestiona en cada tick
   if(InpTrailingPoints > 0)
      ManageTrailing();

   // la senal solo se evalua una vez por vela nueva
   datetime barTime = iTime(_Symbol, InpTimeframe, 0);
   if(barTime == g_lastBarTime)
      return;
   g_lastBarTime = barTime;

   int signal = GetCrossSignal();   // +1 cruce alcista, -1 bajista, 0 nada
   if(signal == 0)
      return;

   ulong ticket = FindMyPosition();

   if(ticket > 0)
     {
      long type = PositionGetInteger(POSITION_TYPE);
      bool opposite = (signal > 0 && type == POSITION_TYPE_SELL) ||
                      (signal < 0 && type == POSITION_TYPE_BUY);
      if(!opposite)
         return;                     // ya estamos en la direccion del cruce
      if(!InpCloseOnOpposite)
         return;
      if(!g_trade.PositionClose(ticket))
        {
         Print("No pude cerrar la posicion #", ticket, ": ", g_trade.ResultRetcodeDescription());
         return;
        }
      Print("Posicion #", ticket, " cerrada por cruce contrario");
      if(!InpOpenOnOpposite)
         return;
     }

   if(signal > 0)
      OpenPosition(ORDER_TYPE_BUY);
   else
      OpenPosition(ORDER_TYPE_SELL);
  }

//+------------------------------------------------------------------+
//| Senal: cruce del cierre de vela con la EMA (velas 1 y 2)         |
//+------------------------------------------------------------------+
int GetCrossSignal()
  {
   double ema[2], close[2];
   ArraySetAsSeries(ema, true);
   ArraySetAsSeries(close, true);
   // indice 0 del array = vela cerrada mas reciente (shift 1)
   if(CopyBuffer(g_emaHandle, 0, 1, 2, ema) != 2)
      return 0;
   if(CopyClose(_Symbol, InpTimeframe, 1, 2, close) != 2)
      return 0;

   bool crossUp   = (close[1] <= ema[1] && close[0] > ema[0]);
   bool crossDown = (close[1] >= ema[1] && close[0] < ema[0]);

   if(crossUp)   return  1;
   if(crossDown) return -1;
   return 0;
  }

//+------------------------------------------------------------------+
//| Abrir posicion con SL/TP en puntos                               |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE type)
  {
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double price  = (type == ORDER_TYPE_BUY)
                   ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                   : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // distancia minima que exige el broker para SL/TP
   long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);

   double sl = 0.0, tp = 0.0;
   if(InpStopLossPoints > 0)
     {
      long slPts = MathMax((long)InpStopLossPoints, stopsLevel);
      sl = (type == ORDER_TYPE_BUY) ? price - slPts * point : price + slPts * point;
      sl = NormalizeDouble(sl, digits);
     }
   if(InpTakeProfitPoints > 0)
     {
      long tpPts = MathMax((long)InpTakeProfitPoints, stopsLevel);
      tp = (type == ORDER_TYPE_BUY) ? price + tpPts * point : price - tpPts * point;
      tp = NormalizeDouble(tp, digits);
     }

   double lots = NormalizeLots(InpLots);
   if(lots <= 0.0)
     {
      Print("Lote invalido tras normalizar: ", InpLots);
      return;
     }

   bool ok = (type == ORDER_TYPE_BUY)
             ? g_trade.Buy(lots, _Symbol, 0.0, sl, tp, InpTradeComment)
             : g_trade.Sell(lots, _Symbol, 0.0, sl, tp, InpTradeComment);

   if(ok)
      Print((type == ORDER_TYPE_BUY ? "COMPRA" : "VENTA"), " abierta | lote ",
            DoubleToString(lots, 2), " | SL ", DoubleToString(sl, digits),
            " | TP ", DoubleToString(tp, digits));
   else
      Print("Fallo al abrir: ", g_trade.ResultRetcodeDescription());
  }

//+------------------------------------------------------------------+
//| Trailing stop simple                                             |
//+------------------------------------------------------------------+
void ManageTrailing()
  {
   ulong ticket = FindMyPosition();
   if(ticket == 0)
      return;

   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   long   type   = PositionGetInteger(POSITION_TYPE);
   double curSL  = PositionGetDouble(POSITION_SL);
   double curTP  = PositionGetDouble(POSITION_TP);

   double newSL;
   if(type == POSITION_TYPE_BUY)
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      newSL = NormalizeDouble(bid - InpTrailingPoints * point, digits);
      // solo mover a favor y respetando el paso minimo
      if(newSL > PositionGetDouble(POSITION_PRICE_OPEN) &&
         (curSL == 0.0 || newSL >= curSL + InpTrailingStepPts * point))
         g_trade.PositionModify(ticket, newSL, curTP);
     }
   else
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      newSL = NormalizeDouble(ask + InpTrailingPoints * point, digits);
      if(newSL < PositionGetDouble(POSITION_PRICE_OPEN) &&
         (curSL == 0.0 || newSL <= curSL - InpTrailingStepPts * point))
         g_trade.PositionModify(ticket, newSL, curTP);
     }
  }

//+------------------------------------------------------------------+
//| Busca la posicion de ESTE EA en ESTE simbolo (0 si no hay)       |
//+------------------------------------------------------------------+
ulong FindMyPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
         return ticket;
     }
   return 0;
  }

//+------------------------------------------------------------------+
//| Ajusta el lote a los limites y paso del broker                   |
//+------------------------------------------------------------------+
double NormalizeLots(double lots)
  {
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep > 0.0)
      lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return NormalizeDouble(lots, 2);
  }
//+------------------------------------------------------------------+
