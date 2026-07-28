#!/bin/bash
echo "== can VM reach Azure OpenAI endpoint? =="
curl -sS -m 10 -o /dev/null -w "nikos_https=%{http_code}\n" https://nikos-m6v5aj9v-eastus2.cognitiveservices.azure.com/ 2>&1 | tail -1
echo "== can VM reach management/token endpoint? =="
curl -sS -m 10 -o /dev/null -w "login=%{http_code}\n" https://login.microsoftonline.com/ 2>&1 | tail -1
echo "== general internet =="
curl -sS -m 10 -o /dev/null -w "bing=%{http_code}\n" https://www.bing.com/ 2>&1 | tail -1
echo DONE
